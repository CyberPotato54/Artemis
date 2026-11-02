//! The reuse tree's pure side: the tokens behind the live KV, the sizes a snapshot takes,
//! where a prefill splits to take snapshots, which state a request restores from, and which KV ranges the restore
//! copies. Nothing here touches the GPU or the pool; `Snapshots` (host RAM) drives it.
use crate::session::state_sizes;
use crate::{Cut, Source};
use qwen36::dims::Dims;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

/// Interval snapshots land on every multiple of this many tokens.
pub const INTERVAL: usize = 4096;

/// The length of the common prefix of `a` and `b`.
pub fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The tokens behind the live KV cache. A call that writes from position `a` truncates the record to `a` and,
/// on success, extends it by its tokens; a call that fails past validation leaves it truncated at `a`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvRecord {
    tokens: Vec<u32>,
}

impl KvRecord {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// A write of `tokens` from position `at` succeeded.
    pub fn wrote(&mut self, at: usize, tokens: &[u32]) {
        self.tokens.truncate(at);
        self.tokens.extend_from_slice(tokens);
    }

    /// A write from position `at` failed: the KV from there on is undefined.
    pub fn failed(&mut self, at: usize) {
        self.tokens.truncate(at);
    }

    /// A restore: the KV now holds exactly these tokens.
    pub fn set(&mut self, tokens: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
    }
}

/// What a snapshot costs, from the model's dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    /// Bytes of KV one position takes across all buffers.
    pub kv_per_pos: u64,
    /// K and V of every attention layer.
    pub kv_buffers: usize,
    /// Bytes of DeltaNet conv and recurrent state.
    pub state: u64,
    /// Copies a state takes: the conv and the recurrent state of every DeltaNet layer.
    pub state_slices: usize,
}

impl Sizes {
    pub fn of(d: &Dims) -> Sizes {
        // `state_sizes` at one position: the KV of one position in one buffer, and the per-layer state.
        let (kv, conv, rec, _) = state_sizes(d, 1);
        let attention = (0..d.n_layer).filter(|&l| d.is_attention(l)).count();
        let delta = d.n_layer - attention;
        Sizes {
            kv_per_pos: kv * 2 * attention as u64,
            kv_buffers: 2 * attention,
            state: (conv + rec) * delta as u64,
            state_slices: 2 * delta,
        }
    }

    /// Bytes of one position in one KV buffer.
    pub fn kv_buffer_per_pos(&self) -> u64 {
        self.kv_per_pos / self.kv_buffers as u64
    }

    /// Bytes of the KV segment `[from, to)` across all buffers.
    pub fn segment(&self, from: usize, to: usize) -> u64 {
        to.saturating_sub(from) as u64 * self.kv_per_pos
    }
}

/// Why a position is a snapshot point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Marks {
    pub interval: bool,
    pub prompt_end: bool,
    pub user_query: bool,
    pub reply_end: bool,
}

impl Marks {
    pub const INTERVAL: Marks = Marks {
        interval: true,
        prompt_end: false,
        user_query: false,
        reply_end: false,
    };
    pub const PROMPT_END: Marks = Marks {
        interval: false,
        prompt_end: true,
        user_query: false,
        reply_end: false,
    };
    /// A prompt end that is also the last real user query.
    pub const USER_QUERY: Marks = Marks {
        interval: false,
        prompt_end: true,
        user_query: true,
        reply_end: false,
    };
    pub const REPLY_END: Marks = Marks {
        interval: false,
        prompt_end: false,
        user_query: false,
        reply_end: true,
    };

    pub fn union(self, o: Marks) -> Marks {
        Marks {
            interval: self.interval || o.interval,
            prompt_end: self.prompt_end || o.prompt_end,
            user_query: self.user_query || o.user_query,
            reply_end: self.reply_end || o.reply_end,
        }
    }

    /// The restore source these marks name: `UserQuery` over `PromptEnd` over `ReplyEnd` over `Interval`.
    pub fn source(self) -> Source {
        if self.user_query {
            Source::UserQuery
        } else if self.prompt_end {
            Source::PromptEnd
        } else if self.reply_end {
            Source::ReplyEnd
        } else if self.interval {
            Source::Interval
        } else {
            Source::None
        }
    }

    pub fn is_empty(self) -> bool {
        self == Marks::default()
    }
}

/// Where a prefill of `prompt[from..n]` splits: every multiple of `INTERVAL` in `(from, n]` and the cut's
/// `at` when `from < at <= n`, then `n`. Each is a segment end with the marks its snapshot takes; `n` carries none
/// unless it is a point itself.
pub fn split_points(from: usize, n: usize, cut: Option<Cut>) -> Vec<(usize, Marks)> {
    let mut ends: BTreeMap<usize, Marks> = BTreeMap::new();
    if n <= from {
        return Vec::new();
    }
    let mut at = (from / INTERVAL + 1) * INTERVAL;
    while at <= n {
        ends.insert(at, Marks::INTERVAL);
        at += INTERVAL;
    }
    if let Some(c) = cut {
        let at = c.at as usize;
        if from < at && at <= n {
            let m = if c.user_query {
                Marks::USER_QUERY
            } else {
                Marks::PROMPT_END
            };
            let e = ends.entry(at).or_default();
            *e = e.union(m);
        }
    }
    ends.entry(n).or_default();
    ends.into_iter().collect()
}

/// A snapshot's handle in the tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapId(pub u32);

/// One snapshot: its full token list (a prefix of its descendants'), its place and its bookkeeping.
#[derive(Clone, Debug)]
pub struct Node {
    pub tokens: Vec<u32>,
    /// The deepest node whose tokens are a prefix of this node's; `None` is the root at position 0.
    pub parent: Option<SnapId>,
    pub marks: Marks,
    /// Whether the node still holds its DeltaNet state (a demoted node keeps its KV segment).
    pub has_state: bool,
    /// The tick the snapshot was taken at.
    pub taken: u64,
    pub last_used: u64,
    pub children: u32,
}

impl Node {
    pub fn pos(&self) -> usize {
        self.tokens.len()
    }
}

/// The state a request restores from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice {
    pub source: Source,
    /// The snapshot, when the source is one.
    pub snap: Option<SnapId>,
    /// Where the prefill resumes.
    pub pos: usize,
    /// The longest common prefix of the prompt with any node or the live tokens.
    pub diverged_at: usize,
}

/// What a snapshot at some tokens needs of the tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// A node with these tokens holds a state already: touch it.
    Touch(SnapId),
    /// A stateless node has these tokens: it gets a state back.
    AddState(SnapId),
    /// A new node whose KV segment starts at `from`, the parent's position.
    New { parent: Option<SnapId>, from: usize },
}

/// Why a snapshot was not taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    ReuseOff,
    TooBig,
    NoRoom,
}

/// The snapshot tree, with the bytes it holds against its limit.
pub struct Tree {
    nodes: BTreeMap<SnapId, Node>,
    next_id: u32,
    /// The next tick to hand out.
    tick: u64,
    limit: u64,
    used: u64,
    sizes: Sizes,
    counters: ReuseStats,
}

impl Tree {
    pub fn new(limit: u64, sizes: Sizes) -> Tree {
        Tree {
            nodes: BTreeMap::new(),
            next_id: 0,
            tick: 1,
            limit,
            used: 0,
            sizes,
            counters: ReuseStats::default(),
        }
    }

    pub fn sizes(&self) -> Sizes {
        self.sizes
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: SnapId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn ids(&self) -> Vec<SnapId> {
        self.nodes.keys().copied().collect()
    }

    fn next_tick(&mut self) -> u64 {
        let t = self.tick;
        self.tick += 1;
        t
    }

    /// The position of `parent`, or 0 for the root.
    fn pos_of(&self, parent: Option<SnapId>) -> usize {
        parent.and_then(|p| self.nodes.get(&p)).map_or(0, Node::pos)
    }

    /// The state a request for `prompt` restores from: the longest of the live session (when it holds tokens) and
    /// every node with a state, each a proper prefix of the prompt; a tie goes to the live session; none is a reset.
    pub fn choose(&self, prompt: &[u32], live: Option<&[u32]>) -> Choice {
        let mut diverged_at = 0;
        let mut best = Choice {
            source: Source::None,
            snap: None,
            pos: 0,
            diverged_at: 0,
        };
        if let Some(live) = live.filter(|l| !l.is_empty()) {
            let l = lcp(prompt, live);
            diverged_at = l;
            if l == live.len() && l < prompt.len() {
                best = Choice {
                    source: Source::Live,
                    snap: None,
                    pos: l,
                    diverged_at: 0,
                };
            }
        }
        for (&id, node) in &self.nodes {
            let l = lcp(prompt, &node.tokens);
            diverged_at = diverged_at.max(l);
            let proper = l == node.pos() && l < prompt.len();
            // Strictly longer: the live session and earlier nodes keep a tie.
            if node.has_state && proper && l > best.pos {
                best = Choice {
                    source: node.marks.source(),
                    snap: Some(id),
                    pos: l,
                    diverged_at: 0,
                };
            }
        }
        best.diverged_at = diverged_at;
        best
    }

    /// What a snapshot at `tokens` needs: a touch, a state for a stateless node, or a new node under the deepest
    /// node (stateless ones included) whose tokens are a proper prefix.
    pub fn plan(&self, tokens: &[u32]) -> Plan {
        let mut parent: Option<SnapId> = None;
        let mut depth = 0;
        for (&id, node) in &self.nodes {
            if node.tokens == tokens {
                return if node.has_state {
                    Plan::Touch(id)
                } else {
                    Plan::AddState(id)
                };
            }
            if node.pos() < tokens.len() && tokens.starts_with(&node.tokens) && node.pos() > depth {
                parent = Some(id);
                depth = node.pos();
            }
        }
        Plan::New {
            parent,
            from: depth,
        }
    }

    /// The bytes `plan` adds for a snapshot at position `pos`.
    pub fn need(&self, plan: Plan, pos: usize) -> u64 {
        match plan {
            Plan::Touch(_) => 0,
            Plan::AddState(_) => self.sizes.state,
            Plan::New { from, .. } => self.sizes.state + self.sizes.segment(from, pos),
        }
    }

    /// Adds a node with a state under `parent`.
    pub fn insert(&mut self, tokens: &[u32], parent: Option<SnapId>, marks: Marks) -> SnapId {
        let t = self.next_tick();
        self.add_node(tokens, parent, marks, true, t, t)
    }

    /// Adds a node as it was parked, with its own ticks and state flag. The tick counter moves above both ticks.
    pub fn insert_parked(
        &mut self,
        tokens: &[u32],
        parent: Option<SnapId>,
        marks: Marks,
        has_state: bool,
        taken: u64,
        last_used: u64,
    ) -> SnapId {
        self.tick = self.tick.max(taken.max(last_used) + 1);
        self.add_node(tokens, parent, marks, has_state, taken, last_used)
    }

    fn add_node(
        &mut self,
        tokens: &[u32],
        parent: Option<SnapId>,
        marks: Marks,
        has_state: bool,
        taken: u64,
        last_used: u64,
    ) -> SnapId {
        // A plain `assert!`: a parent that isn't here would count as the root, and release runs skip debug assertions.
        assert!(
            parent.is_none_or(|p| self.nodes.contains_key(&p)),
            "the parent {parent:?} is not in the tree (parents come before children)"
        );
        let id = SnapId(self.next_id);
        self.next_id += 1;
        self.used += self.sizes.segment(self.pos_of(parent), tokens.len());
        if has_state {
            self.used += self.sizes.state;
        }
        if let Some(p) = parent.and_then(|p| self.nodes.get_mut(&p)) {
            p.children += 1;
        }
        self.nodes.insert(
            id,
            Node {
                tokens: tokens.to_vec(),
                parent,
                marks,
                has_state,
                taken,
                last_used,
                children: 0,
            },
        );
        id
    }

    /// The node was used (a restore, or a snapshot landing on it again): marks unioned, `last_used` bumped.
    pub fn touch(&mut self, id: SnapId, marks: Marks) {
        let t = self.next_tick();
        if let Some(n) = self.nodes.get_mut(&id) {
            n.marks = n.marks.union(marks);
            n.last_used = t;
        }
    }

    /// A snapshot landed on a stateless node: it holds a state again.
    pub fn add_state(&mut self, id: SnapId, marks: Marks) {
        let t = self.next_tick();
        let state = self.sizes.state;
        if let Some(n) = self.nodes.get_mut(&id) {
            if !n.has_state {
                n.has_state = true;
                n.taken = t;
                self.used += state;
            }
            n.marks = n.marks.union(marks);
            n.last_used = t;
        }
    }

    /// The nodes from the root to `id`, each with its KV segment `[from, to)`.
    pub fn chain(&self, id: SnapId) -> Vec<(SnapId, usize, usize)> {
        let mut out = Vec::new();
        let mut cur = Some(id);
        while let Some(c) = cur {
            let Some(n) = self.nodes.get(&c) else { break };
            out.push((c, self.pos_of(n.parent), n.pos()));
            cur = n.parent;
        }
        out.reverse();
        out
    }
}

/// The parts of the chain's segments at or above `from`, in order, empty parts left out: what a restore copies when
/// the live KV already holds the first `from` positions.
pub fn kv_copies(chain: &[(SnapId, usize, usize)], from: usize) -> Vec<(SnapId, usize, usize)> {
    chain
        .iter()
        .filter_map(|&(id, a, b)| {
            let lo = a.max(from);
            (lo < b).then_some((id, lo, b))
        })
        .collect()
}

/// A piece of the pool the tree no longer holds: the pool releases it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Freed {
    /// The node's DeltaNet state.
    State(SnapId),
    /// The node itself: its KV segment (it holds no state by then).
    Node(SnapId),
}

/// The reuse counters. The tree fills all but `reused_tokens` and `prefilled_tokens` (the worker's)
/// and `pool_cut` and `chunks_refused` (the pool's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReuseStats {
    pub snapshots_held: u64,
    pub pool_bytes_used: u64,
    pub pool_bytes_limit: u64,
    pub restores: u64,
    pub restore_bytes: u64,
    pub reused_tokens: u64,
    pub prefilled_tokens: u64,
    pub evicted_protected: u64,
    pub evicted_unprotected: u64,
    pub skipped: u64,
    pub pool_cut: bool,
    /// The pool chunks the driver refused (each later miss tries again, so one refusal can count many times).
    pub chunks_refused: u64,
}

impl Tree {
    /// The protected set and the part of it exempt while `running` runs. Heads are the `user_query` nodes
    /// that no other `user_query` node's tokens extend. Protected: each head with a state, and per head the newest
    /// (`taken`) `reply_end` node with a state whose tokens extend it. Exempt: the running request's head (the deepest
    /// head that is a prefix of `running`), its protected reply end, and every protected node that is a prefix of
    /// `running`.
    pub fn protected(&self, running: &[u32]) -> (BTreeSet<SnapId>, BTreeSet<SnapId>) {
        let heads: Vec<SnapId> = self
            .nodes
            .iter()
            .filter(|(_, n)| {
                n.marks.user_query
                    && !self.nodes.values().any(|m| {
                        m.marks.user_query && m.pos() > n.pos() && m.tokens.starts_with(&n.tokens)
                    })
            })
            .map(|(&id, _)| id)
            .collect();
        let mut protected = BTreeSet::new();
        let mut reply_of = BTreeMap::new();
        for &h in &heads {
            let head = &self.nodes[&h];
            if head.has_state {
                protected.insert(h);
            }
            let newest = self
                .nodes
                .iter()
                .filter(|(_, r)| {
                    r.marks.reply_end && r.has_state && r.tokens.starts_with(&head.tokens)
                })
                .max_by_key(|(_, r)| r.taken);
            if let Some((&r, _)) = newest {
                protected.insert(r);
                reply_of.insert(h, r);
            }
        }
        let mut exempt = BTreeSet::new();
        let running_head = heads
            .iter()
            .filter(|h| running.starts_with(&self.nodes[h].tokens))
            .max_by_key(|h| self.nodes[h].pos());
        if let Some(h) = running_head {
            if protected.contains(h) {
                exempt.insert(*h);
            }
            exempt.extend(reply_of.get(h));
        }
        exempt.extend(
            protected
                .iter()
                .filter(|p| running.starts_with(&self.nodes[p].tokens)),
        );
        (protected, exempt)
    }

    /// Frees `need` bytes for a snapshot, or says why it can't be had. Stateless childless nodes not on `running` go
    /// first (the sweep); then unprotected nodes lose their states, least recently used first; then protected,
    /// non-exempt ones, the same way. The pool releases what the list names, in order. A failed call changes
    /// nothing: the list would be lost with the error, so every step is undone. It counts no skip itself.
    pub fn make_room(&mut self, need: u64, running: &[u32]) -> Result<Vec<Freed>, Skip> {
        if self.limit == 0 {
            return Err(Skip::ReuseOff);
        }
        if need > self.limit {
            return Err(Skip::TooBig);
        }
        let mut freed = Vec::new();
        let mut removed = Vec::new();
        self.sweep_into(running, &mut freed, &mut removed);
        let (protected, exempt) = self.protected(running);
        let (mut cut_unprotected, mut cut_protected) = (0, 0);
        while self.used + need > self.limit {
            let victim = self
                .lru_with_state(|id| !protected.contains(&id))
                .map(|id| (id, false))
                .or_else(|| {
                    self.lru_with_state(|id| protected.contains(&id) && !exempt.contains(&id))
                        .map(|id| (id, true))
                });
            let Some((id, was_protected)) = victim else {
                self.undo(&freed, removed);
                return Err(Skip::NoRoom);
            };
            self.demote(id);
            freed.push(Freed::State(id));
            self.cascade(Some(id), running, &mut freed, &mut removed);
            if was_protected {
                cut_protected += 1;
            } else {
                cut_unprotected += 1;
            }
        }
        self.counters.evicted_protected += cut_protected;
        self.counters.evicted_unprotected += cut_unprotected;
        Ok(freed)
    }

    /// `make_room(need)`, then `alloc` with what that freed: the caller releases it and tries to allocate its slices
    /// (`Some` when it got them). The tree counts exact bytes and the pool hands out aligned runs, so `alloc` can find
    /// no run though the tree has room. Each of up to `retries` retries then demotes one more state, the next least
    /// recently used under `make_room`'s rules, by asking for more than is free now, and calls `alloc` again.
    /// `Ok(None)` when `make_room` refuses (at any attempt) or the retries are spent; what was freed stays freed.
    pub fn make_room_and_alloc<T, E>(
        &mut self,
        need: u64,
        running: &[u32],
        retries: u64,
        mut alloc: impl FnMut(&[Freed]) -> Result<Option<T>, E>,
    ) -> Result<Option<T>, E> {
        let mut asking = need;
        for _ in 0..=retries {
            let Ok(freed) = self.make_room(asking, running) else {
                return Ok(None);
            };
            if let Some(got) = alloc(&freed)? {
                return Ok(Some(got));
            }
            asking = self.limit.saturating_sub(self.used) + 1;
        }
        Ok(None)
    }

    /// Demotes one node: it loses its state. Then stateless childless nodes are freed, up its chain, except those
    /// on `running` (the new snapshot's chain), which the next `make_room` sweeps.
    pub fn drop_state(&mut self, id: SnapId, running: &[u32]) -> Vec<Freed> {
        let mut freed = Vec::new();
        if self.nodes.get(&id).is_some_and(|n| n.has_state) {
            self.demote(id);
            freed.push(Freed::State(id));
        }
        self.cascade(Some(id), running, &mut freed, &mut Vec::new());
        freed
    }

    /// Frees every stateless childless node not on `running`, and the parents that leaves childless.
    pub fn sweep(&mut self, running: &[u32]) -> Vec<Freed> {
        let mut freed = Vec::new();
        self.sweep_into(running, &mut freed, &mut Vec::new());
        freed
    }

    /// A snapshot was not taken (a `Skip`, or a refused pool allocation).
    pub fn count_skip(&mut self) {
        self.counters.skipped += 1;
    }

    /// A restore copied `bytes` (the state and the KV past the common prefix).
    pub fn count_restore(&mut self, bytes: u64) {
        self.counters.restores += 1;
        self.counters.restore_bytes += bytes;
    }

    /// The counters, with the held snapshots and the pool's bytes read off the tree. `reused_tokens` and
    /// `prefilled_tokens` stay 0 here (the worker fills them) and so do `pool_cut` and `chunks_refused` (the pool's).
    pub fn stats(&self) -> ReuseStats {
        ReuseStats {
            snapshots_held: self.nodes.values().filter(|n| n.has_state).count() as u64,
            pool_bytes_used: self.used,
            pool_bytes_limit: self.limit,
            ..self.counters
        }
    }

    /// The least recently used node with a state among those `pick` admits. Ticks are unique, so no id breaks a tie.
    fn lru_with_state(&self, pick: impl Fn(SnapId) -> bool) -> Option<SnapId> {
        self.nodes
            .iter()
            .filter(|(&id, n)| n.has_state && pick(id))
            .min_by_key(|(_, n)| n.last_used)
            .map(|(&id, _)| id)
    }

    fn demote(&mut self, id: SnapId) {
        if let Some(n) = self.nodes.get_mut(&id) {
            if n.has_state {
                n.has_state = false;
                self.used -= self.sizes.state;
            }
        }
    }

    /// Removes a stateless node and its segment, handing the node back so a failed `make_room` can restore it.
    fn free_node(&mut self, id: SnapId) -> Node {
        let node = self.nodes.remove(&id).expect("a node to free");
        self.used -= self.sizes.segment(self.pos_of(node.parent), node.pos());
        if let Some(p) = node.parent.and_then(|p| self.nodes.get_mut(&p)) {
            p.children -= 1;
        }
        node
    }

    /// Frees `from` if it is stateless, childless and off `running`, then its parent by the same rule, and so on.
    fn cascade(
        &mut self,
        from: Option<SnapId>,
        running: &[u32],
        freed: &mut Vec<Freed>,
        removed: &mut Vec<Node>,
    ) {
        let mut cur = from;
        while let Some(c) = cur {
            let Some(n) = self.nodes.get(&c) else { break };
            if n.has_state || n.children > 0 || running.starts_with(&n.tokens) {
                break;
            }
            cur = n.parent;
            removed.push(self.free_node(c));
            freed.push(Freed::Node(c));
        }
    }

    fn sweep_into(&mut self, running: &[u32], freed: &mut Vec<Freed>, removed: &mut Vec<Node>) {
        let mut leaves: Vec<(u64, SnapId)> = self
            .nodes
            .iter()
            .filter(|(_, n)| !n.has_state && n.children == 0 && !running.starts_with(&n.tokens))
            .map(|(&id, n)| (n.last_used, id))
            .collect();
        leaves.sort_unstable();
        for (_, id) in leaves {
            self.cascade(Some(id), running, freed, removed);
        }
    }

    /// Puts back what `make_room` took, newest first. `removed` holds the freed nodes in the order they went.
    fn undo(&mut self, freed: &[Freed], mut removed: Vec<Node>) {
        for f in freed.iter().rev() {
            match *f {
                Freed::State(id) => {
                    self.nodes.get_mut(&id).expect("a demoted node").has_state = true;
                    self.used += self.sizes.state;
                }
                Freed::Node(id) => {
                    let node = removed.pop().expect("a freed node");
                    self.used += self.sizes.segment(self.pos_of(node.parent), node.pos());
                    if let Some(p) = node.parent.and_then(|p| self.nodes.get_mut(&p)) {
                        p.children += 1;
                    }
                    self.nodes.insert(id, node);
                }
            }
        }
    }
}

/// Which parked nodes to bring back, and in what order. `nodes[i]` is `(parent index, last_used, own bytes)`, parents
/// listed before children. The newest come back first, each with the ancestors it lacks (parents first), while the
/// total fits `limit`; a node whose chain doesn't fit is skipped and older ones are still tried.
pub fn unpark_order(nodes: &[(Option<usize>, u64, u64)], limit: u64) -> Vec<usize> {
    let mut by_age: Vec<usize> = (0..nodes.len()).collect();
    by_age.sort_by_key(|&i| (Reverse(nodes[i].1), i));
    let mut back = vec![false; nodes.len()];
    let mut order = Vec::new();
    let mut total = 0u64;
    for i in by_age {
        let mut chain = Vec::new();
        let mut cur = Some(i);
        while let Some(c) = cur.filter(|&c| !back[c]) {
            chain.push(c);
            cur = nodes[c].0;
        }
        let bytes: u64 = chain.iter().map(|&c| nodes[c].2).sum();
        if chain.is_empty() || total.saturating_add(bytes) > limit {
            continue;
        }
        total += bytes;
        for &c in chain.iter().rev() {
            back[c] = true;
            order.push(c);
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cut, Source};
    use qwen36::dims::Dims;

    fn t(r: std::ops::Range<u32>) -> Vec<u32> {
        r.collect()
    }

    #[test]
    fn sizes_on_qwen36_are_20480_bytes_a_position_and_62_81_mib_of_state() {
        let s = Sizes::of(&Dims::QWEN36);
        assert_eq!(
            (s.kv_per_pos, s.kv_buffers, s.state, s.state_slices),
            (20_480, 20, 65_863_680, 60)
        );
        assert_eq!(s.kv_buffer_per_pos(), 1_024);
        assert_eq!(s.segment(0, 30_000), 614_400_000);
        let tiny = Sizes::of(&Dims::tiny());
        assert_eq!(
            (tiny.kv_per_pos, tiny.kv_buffers, tiny.state_slices),
            (256, 2, 6)
        );
    }

    #[test]
    fn the_long_prompt_splits_at_six_intervals_and_the_cut() {
        let p = split_points(
            0,
            27_416,
            Some(Cut {
                at: 27_410,
                user_query: true,
            }),
        );
        let ends: Vec<usize> = p.iter().map(|x| x.0).collect();
        assert_eq!(
            ends,
            vec![4096, 8192, 12288, 16384, 20480, 24576, 27410, 27416]
        );
        assert!(p[..6].iter().all(|x| x.1 == Marks::INTERVAL));
        assert_eq!(p[6].1, Marks::USER_QUERY);
        assert!(p[7].1.is_empty());
    }

    #[test]
    fn split_points_start_after_the_restore_and_merge_a_cut_on_an_interval() {
        let p = split_points(
            5_000,
            9_000,
            Some(Cut {
                at: 8_192,
                user_query: false,
            }),
        );
        assert_eq!(
            p,
            vec![
                (8_192, Marks::INTERVAL.union(Marks::PROMPT_END)),
                (9_000, Marks::default())
            ]
        );
        assert_eq!(
            split_points(
                100,
                200,
                Some(Cut {
                    at: 100,
                    user_query: true
                })
            ),
            vec![(200, Marks::default())]
        );
        assert_eq!(split_points(0, 4_096, None), vec![(4_096, Marks::INTERVAL)]);
    }

    #[test]
    fn a_candidate_must_leave_at_least_one_token() {
        let mut tr = Tree::new(1 << 40, Sizes::of(&Dims::QWEN36));
        let a = tr.insert(&t(0..100), None, Marks::PROMPT_END);
        let c = tr.choose(&t(0..100), None); // an exact repeat
        assert_eq!(
            (c.source, c.snap, c.pos, c.diverged_at),
            (Source::None, None, 0, 100)
        );
        let c = tr.choose(&t(0..101), None);
        assert_eq!((c.source, c.snap, c.pos), (Source::PromptEnd, Some(a), 100));
    }

    #[test]
    fn the_longest_candidate_wins_and_the_live_session_wins_a_tie() {
        let mut tr = Tree::new(1 << 40, Sizes::of(&Dims::QWEN36));
        let a = tr.insert(&t(0..50), None, Marks::INTERVAL);
        let b = tr.insert(&t(0..80), Some(a), Marks::REPLY_END);
        assert_eq!(tr.choose(&t(0..90), Some(&t(0..60))).snap, Some(b));
        let c = tr.choose(&t(0..90), Some(&t(0..80)));
        assert_eq!((c.source, c.snap, c.pos), (Source::Live, None, 80));
        let mut other = t(0..90);
        other[70] = 999_999; // diverges inside b: only a is a candidate
        let c = tr.choose(&other, None);
        assert_eq!((c.snap, c.pos, c.diverged_at), (Some(a), 50, 70));
    }

    #[test]
    fn an_unusable_live_session_is_never_a_candidate() {
        // The caller passes None
        let tr = Tree::new(1 << 40, Sizes::of(&Dims::QWEN36));
        assert_eq!(tr.choose(&t(0..10), None).source, Source::None);
    }

    #[test]
    fn a_session_with_no_tokens_is_never_a_candidate() {
        // a new or reset session would tie at position 0
        let mut tr = Tree::new(1 << 40, Sizes::of(&Dims::QWEN36));
        let c = tr.choose(&t(0..10), Some(&[]));
        assert_eq!((c.source, c.snap, c.pos), (Source::None, None, 0));
        let a = tr.insert(&t(0..5), None, Marks::INTERVAL);
        let c = tr.choose(&t(0..10), Some(&[]));
        assert_eq!((c.source, c.snap, c.pos), (Source::Interval, Some(a), 5));
    }

    #[test]
    fn insert_parked_keeps_the_ticks_and_a_stateless_flag() {
        let sizes = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(1 << 40, sizes);
        let a = tr.insert_parked(&t(0..100), None, Marks::INTERVAL, false, 7, 9);
        let b = tr.insert_parked(&t(0..150), Some(a), Marks::REPLY_END, true, 8, 5);
        assert_eq!(
            (
                tr.node(a).unwrap().taken,
                tr.node(a).unwrap().last_used,
                tr.node(a).unwrap().has_state
            ),
            (7, 9, false)
        );
        assert_eq!(
            (tr.node(b).unwrap().taken, tr.node(b).unwrap().last_used),
            (8, 5)
        );
        assert_eq!(
            tr.used(),
            sizes.segment(0, 100) + sizes.segment(100, 150) + sizes.state
        ); // only b holds a state
        let c = tr.insert(&t(0..200), Some(b), Marks::PROMPT_END);
        assert!(tr.node(c).unwrap().taken > 9 && tr.node(c).unwrap().last_used > 9);
        // the counter moved above
    }

    #[test]
    fn a_new_snapshots_parent_is_the_deepest_prefix_and_its_segment_starts_there() {
        let mut tr = Tree::new(1 << 40, Sizes::of(&Dims::QWEN36));
        let a = tr.insert(&t(0..4096), None, Marks::INTERVAL);
        let p = tr.plan(&t(0..5000));
        assert_eq!(
            p,
            Plan::New {
                parent: Some(a),
                from: 4096
            }
        );
        assert_eq!(tr.need(p, 5000), 65_863_680 + 904 * 20_480);
        assert_eq!(tr.plan(&t(0..4096)), Plan::Touch(a));
        let mut side = t(0..5000);
        side[10] = 7;
        assert_eq!(
            tr.plan(&side),
            Plan::New {
                parent: None,
                from: 0
            }
        );
    }

    #[test]
    fn kv_copies_cover_only_past_the_common_prefix() {
        let chain = [
            (SnapId(1), 0, 4096),
            (SnapId(2), 4096, 8192),
            (SnapId(3), 8192, 9000),
        ];
        assert_eq!(
            kv_copies(&chain, 5000),
            vec![(SnapId(2), 5000, 8192), (SnapId(3), 8192, 9000)]
        );
        assert!(kv_copies(&chain, 9000).is_empty());
        assert_eq!(kv_copies(&chain, 0).len(), 3);
    }

    #[test]
    fn the_kv_record_truncates_at_a_write_and_at_a_failed_write() {
        let mut k = KvRecord::default();
        k.wrote(0, &[1, 2, 3, 4]);
        k.wrote(2, &[9]);
        assert_eq!(k.tokens(), &[1, 2, 9]);
        k.failed(1);
        assert_eq!(k.tokens(), &[1]);
        k.set(&[5, 6]);
        assert_eq!(k.tokens(), &[5, 6]);
    }

    #[test]
    fn marks_name_their_source() {
        assert_eq!(Marks::USER_QUERY.source(), Source::UserQuery);
        assert_eq!(
            Marks::PROMPT_END.union(Marks::REPLY_END).source(),
            Source::PromptEnd
        );
        assert_eq!(
            Marks::REPLY_END.union(Marks::INTERVAL).source(),
            Source::ReplyEnd
        );
        assert_eq!(Marks::INTERVAL.source(), Source::Interval);
        assert_eq!(Marks::default().source(), Source::None);
    }

    #[test]
    fn touch_unions_marks_and_bumps_last_used() {
        let mut tr = Tree::new(1 << 40, Sizes::of(&Dims::QWEN36));
        let a = tr.insert(&t(0..10), None, Marks::INTERVAL);
        let b = tr.insert(&t(0..20), Some(a), Marks::REPLY_END);
        let (taken, used) = (tr.node(a).unwrap().taken, tr.node(a).unwrap().last_used);
        tr.touch(a, Marks::PROMPT_END);
        let n = tr.node(a).unwrap();
        assert_eq!(n.marks, Marks::INTERVAL.union(Marks::PROMPT_END));
        assert_eq!(n.taken, taken);
        assert!(n.last_used > used && n.last_used > tr.node(b).unwrap().last_used);
        let before = tr.node(a).unwrap().last_used;
        tr.touch(a, Marks::default());
        assert!(tr.node(a).unwrap().last_used > before);
    }

    #[test]
    fn insert_counts_bytes_and_children() {
        let sizes = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(1 << 40, sizes);
        assert!(tr.is_empty());
        let a = tr.insert(&t(0..4096), None, Marks::INTERVAL);
        assert_eq!(tr.used(), sizes.state + sizes.segment(0, 4096));
        assert_eq!(tr.node(a).unwrap().children, 0);
        let plan = tr.plan(&t(0..5000));
        let need = tr.need(plan, 5000);
        let before = tr.used();
        let b = tr.insert(&t(0..5000), Some(a), Marks::PROMPT_END);
        assert_eq!(tr.used(), before + need);
        assert_eq!(tr.node(a).unwrap().children, 1);
        assert_eq!(tr.node(b).unwrap().children, 0);
        assert_eq!((tr.len(), tr.node(b).unwrap().pos()), (2, 5000));
        assert_eq!(tr.ids(), vec![a, b]);
        assert_eq!((tr.limit(), tr.sizes()), (1 << 40, sizes));
    }

    #[test]
    fn add_state_gives_a_stateless_node_its_state_back() {
        let sizes = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(1 << 40, sizes);
        let a = tr.insert_parked(&t(0..100), None, Marks::INTERVAL, false, 1, 1);
        let p = tr.plan(&t(0..100));
        assert_eq!(p, Plan::AddState(a));
        assert_eq!(tr.need(p, 100), sizes.state);
        let before = tr.used();
        tr.add_state(a, Marks::REPLY_END);
        let n = tr.node(a).unwrap();
        assert!(n.has_state);
        assert_eq!(n.marks, Marks::INTERVAL.union(Marks::REPLY_END));
        assert_eq!(tr.used(), before + sizes.state);
        assert_eq!(tr.plan(&t(0..100)), Plan::Touch(a));
        // the state is new: `taken` restarts (the protected reply end is the one with the newest `taken`)
        assert!(n.taken > 1 && n.taken == n.last_used);
        let taken = n.taken;
        tr.add_state(a, Marks::default());
        assert_eq!(tr.used(), before + sizes.state);
        assert_eq!(tr.node(a).unwrap().taken, taken); // a node that holds a state keeps its `taken`
    }

    #[test]
    fn diverged_at_counts_stateless_nodes_too() {
        let mut tr = Tree::new(1 << 40, Sizes::of(&Dims::QWEN36));
        let a = tr.insert_parked(&t(0..100), None, Marks::INTERVAL, false, 1, 1);
        let mut p = t(0..200);
        p[90] = 999_999;
        let c = tr.choose(&p, None);
        assert_eq!(
            (c.source, c.snap, c.pos, c.diverged_at),
            (Source::None, None, 0, 90)
        );
        // as a parent it still counts in plan
        assert_eq!(
            tr.plan(&t(0..200)),
            Plan::New {
                parent: Some(a),
                from: 100
            }
        );
    }

    #[test]
    fn chain_lists_the_root_first() {
        let mut tr = Tree::new(1 << 40, Sizes::of(&Dims::QWEN36));
        let a = tr.insert(&t(0..4096), None, Marks::INTERVAL);
        let b = tr.insert(&t(0..8192), Some(a), Marks::INTERVAL);
        let c = tr.insert(&t(0..9000), Some(b), Marks::PROMPT_END);
        assert_eq!(
            tr.chain(c),
            vec![(a, 0, 4096), (b, 4096, 8192), (c, 8192, 9000)]
        );
        assert_eq!(tr.chain(a), vec![(a, 0, 4096)]);
        assert!(tr.chain(SnapId(99)).is_empty());
    }

    // ---- protection, eviction, parking order, counters ----

    const SUFFIX: [u32; 4] = [9000, 9001, 9002, 9003]; // <|im_start|>assistant\n<think>\n
    const UNECHOED: [u32; 4] = [9002, 9005, 9006, 9007]; // <think>\n\n</think>\n\n
    const CLOSE: u32 = 9010; // </think>\n\n
    const END: u32 = 9011; // <|im_end|>
    const NL: u32 = 9012;

    fn sys() -> Vec<u32> {
        t(1000..1100)
    }

    fn big() -> Tree {
        Tree::new(1 << 40, Sizes::of(&Dims::QWEN36))
    }

    /// The node holding exactly these tokens.
    fn find(tr: &Tree, tokens: &[u32]) -> Option<SnapId> {
        tr.ids()
            .into_iter()
            .find(|&id| tr.node(id).unwrap().tokens == tokens)
    }

    /// Every chain is whole: it starts at 0, its segments are contiguous and it ends at the node.
    fn assert_chains_whole(tr: &Tree) {
        for id in tr.ids() {
            let chain = tr.chain(id);
            assert_eq!(chain[0].1, 0, "{id:?}'s chain does not reach the root");
            assert!(chain.windows(2).all(|w| w[0].2 == w[1].1));
            let last = chain.last().unwrap();
            assert_eq!((last.0, last.2), (id, tr.node(id).unwrap().pos()));
        }
    }

    /// One snapshot of the live tokens, as the worker takes it.
    fn snap(tr: &mut Tree, tokens: &[u32], marks: Marks) {
        match tr.plan(tokens) {
            Plan::Touch(id) => tr.touch(id, marks),
            p @ Plan::AddState(id) => {
                let need = tr.need(p, tokens.len());
                match tr.make_room(need, tokens) {
                    Ok(_) => tr.add_state(id, marks),
                    Err(_) => tr.count_skip(),
                }
            }
            p @ Plan::New { parent, .. } => {
                let need = tr.need(p, tokens.len());
                match tr.make_room(need, tokens) {
                    Ok(_) => {
                        tr.insert(tokens, parent, marks);
                    }
                    Err(_) => tr.count_skip(),
                }
            }
        }
    }

    /// What the worker does with the tree for one request: choose, touch the restored node, snapshot at every split
    /// point, then the reply end (the state is one token short of the reply).
    fn serve(
        tr: &mut Tree,
        live: &mut Vec<u32>,
        prompt: &[u32],
        cut: Cut,
        reply: &[u32],
    ) -> Choice {
        let choice = tr.choose(prompt, Some(live));
        if let Some(id) = choice.snap {
            tr.touch(id, Marks::default());
        }
        for (end, marks) in split_points(choice.pos, prompt.len(), Some(cut)) {
            if !marks.is_empty() {
                snap(tr, &prompt[..end], marks);
            }
        }
        let mut l = prompt.to_vec();
        l.extend_from_slice(&reply[..reply.len() - 1]);
        snap(tr, &l, Marks::REPLY_END);
        *live = l;
        choice
    }

    /// One assistant step: the model's tokens are `reasoning`, `</think>`, `content`, `<|im_end|>`.
    struct Step {
        reasoning: Vec<u32>,
        content: Vec<u32>,
        tool: Option<Vec<u32>>,
    }

    impl Step {
        fn generated(&self) -> Vec<u32> {
            let mut g = self.reasoning.clone();
            g.push(CLOSE);
            g.extend_from_slice(&self.content);
            g.push(END);
            g
        }
    }

    /// A conversation, rendered as the chat template does: an echoed reasoning re-renders the step exactly;
    /// an unechoed one re-renders four fixed ids in its place; a later user query drops the reasoning of every step
    /// before it.
    struct Conv {
        id: u32,
        echoed: bool,
        queries: Vec<(Vec<u32>, Vec<Step>)>,
        fresh: u32,
    }

    impl Conv {
        fn new(id: u32, echoed: bool) -> Conv {
            Conv {
                id,
                echoed,
                queries: Vec::new(),
                fresh: 0,
            }
        }

        fn fresh(&mut self, n: u32) -> Vec<u32> {
            let at = 100_000 + self.id * 100_000 + self.fresh;
            self.fresh += n;
            t(at..at + n)
        }

        fn query(&mut self) {
            let at = 2_000 + 20_000 * self.id + 100 * self.queries.len() as u32;
            self.queries.push((t(at..at + 20), Vec::new()));
        }

        /// The prompt the next request sends, and its cut.
        fn prompt(&self) -> (Vec<u32>, Cut) {
            let mut p = sys();
            let last = self.queries.len() - 1;
            let (mut at, mut user_query) = (0, false);
            for (qi, (user, steps)) in self.queries.iter().enumerate() {
                p.push(9020);
                p.extend_from_slice(user);
                (at, user_query) = (p.len(), true);
                for s in steps {
                    if self.echoed && qi == last {
                        p.extend_from_slice(&SUFFIX);
                        p.extend(s.generated());
                    } else {
                        p.extend_from_slice(&[9000, 9001]);
                        p.extend_from_slice(&UNECHOED);
                        p.extend_from_slice(&s.content);
                        p.push(END);
                    }
                    p.push(NL);
                    if let Some(tool) = &s.tool {
                        p.extend_from_slice(tool);
                        (at, user_query) = (p.len(), false);
                    }
                }
            }
            p.extend_from_slice(&SUFFIX);
            (
                p,
                Cut {
                    at: at as u32,
                    user_query,
                },
            )
        }

        /// One request; `tool` says whether the reply ends in a tool call (a tool result follows).
        fn turn(&mut self, tr: &mut Tree, live: &mut Vec<u32>, tool: bool) -> Choice {
            let step = Step {
                reasoning: self.fresh(30),
                content: self.fresh(8),
                tool: tool.then(|| self.fresh(20)),
            };
            let (prompt, cut) = self.prompt();
            let choice = serve(tr, live, &prompt, cut, &step.generated());
            self.queries.last_mut().unwrap().1.push(step);
            choice
        }

        /// The position of the reply-end snapshot of the last step.
        fn reply_end_pos(&self, prompt_len: usize) -> usize {
            let g = self.queries.last().unwrap().1.last().unwrap().generated();
            prompt_len + g.len() - 1
        }
    }

    #[test]
    fn a_tool_loop_with_reasoning_echoed_continues_from_the_live_state() {
        let (mut tr, mut live, mut c) = (big(), Vec::new(), Conv::new(0, true));
        c.query();
        let (p1, _) = c.prompt();
        let first = c.turn(&mut tr, &mut live, true);
        assert_eq!((first.source, first.pos), (Source::None, 0));
        let end = c.reply_end_pos(p1.len());
        // the reply-end snapshot and the live state tie at the same position
        let (p2, _) = c.prompt();
        let c_snap = tr.choose(&p2, None);
        assert_eq!((c_snap.source, c_snap.pos), (Source::ReplyEnd, end));
        let second = c.turn(&mut tr, &mut live, false);
        assert_eq!(
            (second.source, second.snap, second.pos),
            (Source::Live, None, end)
        );
    }

    #[test]
    fn a_tool_loop_with_reasoning_echoed_restores_the_reply_end_once_the_live_state_is_gone() {
        let (mut tr, mut live, mut c) = (big(), Vec::new(), Conv::new(0, true));
        c.query();
        let (p1, _) = c.prompt();
        c.turn(&mut tr, &mut live, true);
        let end = c.reply_end_pos(p1.len());
        live.clear(); // an unload, a reset or an error in between
        let second = c.turn(&mut tr, &mut live, false);
        assert_eq!(second.source, Source::ReplyEnd);
        assert_eq!(second.pos, end);
        assert_eq!(tr.node(second.snap.unwrap()).unwrap().pos(), end);
    }

    #[test]
    fn a_tool_loop_without_reasoning_echoed_restores_the_latest_prompt_end() {
        let (mut tr, mut live, mut c) = (big(), Vec::new(), Conv::new(0, false));
        c.query();
        let (_, cut1) = c.prompt();
        c.turn(&mut tr, &mut live, true);
        let (_, cut2) = c.prompt();
        let second = c.turn(&mut tr, &mut live, true);
        assert_eq!(
            (second.source, second.pos),
            (Source::UserQuery, cut1.at as usize)
        );
        let third = c.turn(&mut tr, &mut live, false);
        assert_eq!(
            (third.source, third.pos),
            (Source::PromptEnd, cut2.at as usize)
        );
    }

    #[test]
    fn a_new_user_turn_after_a_tool_loop_restores_the_user_query_snapshot() {
        let (mut tr, mut live, mut c) = (big(), Vec::new(), Conv::new(0, true));
        c.query();
        let (_, loop_cut) = c.prompt();
        for tool in [true, true, false] {
            c.turn(&mut tr, &mut live, tool);
        }
        c.query();
        let next = c.turn(&mut tr, &mut live, false);
        assert_eq!(
            (next.source, next.pos),
            (Source::UserQuery, loop_cut.at as usize)
        );
    }

    #[test]
    fn a_regenerated_reply_restores_the_prompt_end() {
        let (mut tr, mut live) = (big(), Vec::new());
        let mut prompt = sys();
        prompt.extend(t(2000..2020));
        let cut = Cut {
            at: prompt.len() as u32,
            user_query: true,
        };
        prompt.extend_from_slice(&SUFFIX);
        serve(&mut tr, &mut live, &prompt, cut, &t(300_000..300_010));
        let again = serve(&mut tr, &mut live, &prompt, cut, &t(400_000..400_010));
        assert_eq!(
            (again.source, again.pos),
            (Source::UserQuery, cut.at as usize)
        );
    }

    #[test]
    fn two_interleaved_conversations_each_restore_their_own_latest() {
        let (mut tr, mut live) = (big(), Vec::new());
        let (mut a, mut b) = (Conv::new(0, true), Conv::new(1, true));
        a.query();
        b.query();
        let (pa, _) = a.prompt();
        let (pb, _) = b.prompt();
        a.turn(&mut tr, &mut live, true);
        let a_end = a.reply_end_pos(pa.len());
        b.turn(&mut tr, &mut live, true);
        let b_end = b.reply_end_pos(pb.len());
        let a2 = a.turn(&mut tr, &mut live, false);
        assert_eq!((a2.source, a2.pos), (Source::ReplyEnd, a_end));
        let b2 = b.turn(&mut tr, &mut live, false);
        assert_eq!((b2.source, b2.pos), (Source::ReplyEnd, b_end));
    }

    #[test]
    fn a_side_call_between_turns_leaves_the_reply_end_and_copies_kv_past_the_shared_prefix() {
        let (mut tr, mut live) = (big(), Vec::new());
        let mut a = Conv::new(0, true);
        a.query();
        let (pa, _) = a.prompt();
        a.turn(&mut tr, &mut live, true);
        let a_end = a.reply_end_pos(pa.len());
        // a side call that shares only the system prompt
        let mut side = sys();
        side.extend(t(7000..7020));
        let cut = Cut {
            at: side.len() as u32,
            user_query: true,
        };
        side.extend_from_slice(&SUFFIX);
        serve(&mut tr, &mut live, &side, cut, &t(310_000..310_010));
        let side_live = live.clone();
        let a2 = a.turn(&mut tr, &mut live, false);
        assert_eq!((a2.source, a2.pos), (Source::ReplyEnd, a_end));
        let snap_id = a2.snap.unwrap();
        let a1_tokens = tr.node(snap_id).unwrap().tokens.clone();
        let copies = kv_copies(&tr.chain(snap_id), lcp(&side_live, &a1_tokens));
        assert_eq!(copies[0].1, 100);
        assert_eq!(copies.last().unwrap().2, a_end);
        assert!(copies.windows(2).all(|w| w[0].2 == w[1].1));
    }

    #[test]
    fn a_27416_token_first_prompt_takes_six_interval_snapshots_and_the_prompt_end() {
        let (mut tr, mut live) = (big(), Vec::new());
        let prompt = t(0..27_416);
        let cut = Cut {
            at: 27_410,
            user_query: true,
        };
        serve(&mut tr, &mut live, &prompt, cut, &t(500_000..500_010));
        assert_eq!(tr.len(), 8);
        assert_eq!(tr.used(), 8 * 65_863_680 + (27_416 + 9) as u64 * 20_480);
        assert_eq!(tr.stats().snapshots_held, 8);
        assert_chains_whole(&tr);
    }

    #[test]
    fn a_pool_too_small_for_every_protected_snapshot_evicts_other_conversations_first() {
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(4 * s.state + 2_500 * 20_480, s);
        let mut live = Vec::new();
        let mut convs = Vec::new();
        for x in 0..3u32 {
            let prompt = t(10_000 * (x + 1)..10_000 * (x + 1) + 1_000);
            let cut = Cut {
                at: 996,
                user_query: true,
            };
            let reply = t(600_000 + 100 * x..600_010 + 100 * x);
            serve(&mut tr, &mut live, &prompt, cut, &reply);
            let mut end = prompt.clone();
            end.extend_from_slice(&reply[..9]);
            convs.push((prompt[..996].to_vec(), end));
        }
        assert_eq!(tr.stats().skipped, 0);
        let holds =
            |tokens: &[u32]| find(&tr, tokens).is_some_and(|id| tr.node(id).unwrap().has_state);
        for (uq, end) in &convs[..1] {
            // A's head and reply end were both demoted, then the cascade freed both nodes
            assert!(find(&tr, uq).is_none() && find(&tr, end).is_none());
        }
        for (uq, end) in &convs[1..] {
            assert!(holds(uq) && holds(end), "B or C lost a state");
        }
        assert_eq!(tr.stats().evicted_protected, 2);
        assert_eq!(tr.stats().evicted_unprotected, 0);
        assert!(tr.used() <= tr.limit());
        assert_chains_whole(&tr);
    }

    #[test]
    fn unprotected_snapshots_lose_their_state_lru_first() {
        let s = Sizes::of(&Dims::QWEN36);
        let limit = 3 * s.state + s.segment(0, 300);
        let mut tr = Tree::new(limit, s);
        let n1 = tr.insert(&t(0..100), None, Marks::INTERVAL);
        let n2 = tr.insert(&t(0..200), Some(n1), Marks::INTERVAL);
        let n3 = tr.insert(&t(0..300), Some(n2), Marks::INTERVAL);
        tr.touch(n1, Marks::default()); // LRU order: n2, n3, n1
        assert_eq!(tr.used(), limit);
        assert_eq!(tr.make_room(s.state, &[]), Ok(vec![Freed::State(n2)]));
        assert!(!tr.node(n2).unwrap().has_state); // it has a child: it stays
        assert_eq!(tr.used(), limit - s.state);
        // n3 next; it is a leaf, so it goes whole, and n2 (stateless, now a leaf) with it
        assert_eq!(
            tr.make_room(2 * s.state, &[]),
            Ok(vec![Freed::State(n3), Freed::Node(n3), Freed::Node(n2)])
        );
        assert_eq!(tr.len(), 1);
        assert_eq!(tr.used(), s.state + s.segment(0, 100));
        let st = tr.stats();
        assert_eq!((st.evicted_unprotected, st.evicted_protected), (2, 0));
    }

    #[test]
    fn a_pool_with_no_run_and_a_tree_with_room_demotes_the_least_recently_used_and_retries() {
        // The take's retry (`Snapshots::take` runs this method): the tree has room for far more than the new state, but
        // the pool's one chunk holds three states and no run is free. Asking for `need + k × state` would free nothing.
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(10 * s.state, s);
        let ids: Vec<SnapId> = (0..3)
            .map(|i| tr.insert(&t(i * 1000..i * 1000 + 100), None, Marks::INTERVAL))
            .collect();
        let mut pool = crate::FreeList::default();
        pool.add_chunk(3 * s.state);
        let at: BTreeMap<SnapId, (usize, u64)> = ids
            .iter()
            .map(|&id| (id, pool.alloc(s.state).expect("the chunk holds three")))
            .collect();
        assert_eq!(pool.alloc(s.state), None, "no run is free");
        let mut tries = 0;
        let got: Result<_, ()> = tr.make_room_and_alloc(s.state, &[], 4, |freed| {
            tries += 1;
            for f in freed {
                if let Freed::State(id) = f {
                    let (c, o) = at[id];
                    pool.free(c, o, s.state);
                }
            }
            Ok(pool.alloc(s.state))
        });
        assert_eq!(got, Ok(Some((0, 0))), "the oldest state's run was reused");
        assert_eq!(
            tries, 2,
            "the first try found no run; the retry demoted one"
        );
        assert!(!tr.node(ids[0]).is_some_and(|n| n.has_state));
        assert!(ids[1..].iter().all(|&id| tr.node(id).unwrap().has_state));
        assert_eq!(tr.stats().evicted_unprotected, 1);
    }

    #[test]
    fn the_retries_are_bounded_and_end_when_nothing_is_left_to_demote() {
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(10 * s.state, s);
        for i in 0..3 {
            tr.insert(&t(i * 1000..i * 1000 + 100), None, Marks::INTERVAL);
        }
        let mut tries = 0;
        let never = |tries: &mut u32| {
            *tries += 1;
            Ok::<Option<()>, ()>(None)
        };
        // Two retries: three tries, two states demoted.
        assert_eq!(
            tr.make_room_and_alloc(s.state, &[], 2, |_| never(&mut tries)),
            Ok(None)
        );
        assert_eq!((tries, tr.stats().snapshots_held), (3, 1));
        // Many retries: the last state goes, then the tree is empty and the next ask is past the limit.
        assert_eq!(
            tr.make_room_and_alloc(s.state, &[], 9, |_| never(&mut tries)),
            Ok(None)
        );
        assert_eq!((tries, tr.stats().snapshots_held), (5, 0));
    }

    #[test]
    fn stateless_childless_snapshots_are_freed_with_their_kv() {
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = big();
        let a = tr.insert(&t(0..100), None, Marks::INTERVAL);
        let b = tr.insert(&t(0..300), Some(a), Marks::INTERVAL);
        let before = tr.used();
        assert_eq!(tr.drop_state(b, &[]), vec![Freed::State(b), Freed::Node(b)]);
        assert_eq!(tr.used(), before - s.state - s.segment(100, 300));
        assert!(tr.node(b).is_none());
        assert_eq!(tr.node(a).unwrap().children, 0);
        // a stateless leaf left behind is swept, unless it is on the running tokens
        let c = tr.insert_parked(&t(0..150), Some(a), Marks::INTERVAL, false, 50, 50);
        let with_c = tr.used();
        assert_eq!(
            with_c,
            before - s.state - s.segment(100, 300) + s.segment(100, 150)
        );
        assert!(tr.sweep(&t(0..160)).is_empty());
        assert_eq!(tr.used(), with_c);
        assert_eq!(tr.sweep(&[]), vec![Freed::Node(c)]);
        assert_eq!(tr.used(), with_c - s.segment(100, 150));
        // demoting a node on the running tokens leaves it, stateless, for the next sweep
        let d = tr.insert(&t(0..400), Some(a), Marks::INTERVAL);
        assert_eq!(tr.drop_state(d, &t(0..450)), vec![Freed::State(d)]);
        assert!(tr.node(d).is_some());
        assert_eq!(tr.sweep(&t(0..450)), vec![]);
        assert_eq!(tr.sweep(&t(500..510)), vec![Freed::Node(d)]);
        // a demoted parent of a live node stays
        let e = tr.insert(&t(0..500), Some(a), Marks::INTERVAL);
        let f = tr.insert(&t(0..600), Some(e), Marks::INTERVAL);
        assert_eq!(tr.drop_state(e, &[]), vec![Freed::State(e)]);
        assert_eq!(tr.node(e).unwrap().children, 1);
        assert!(tr.node(f).unwrap().has_state);
    }

    #[test]
    fn a_snapshot_bigger_than_the_pool_is_skipped() {
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(500 << 20, s);
        let a = tr.insert(&t(0..100), None, Marks::INTERVAL);
        let need = s.state + s.segment(0, 30_000);
        assert_eq!(tr.make_room(need, &[]), Err(Skip::TooBig));
        assert!(tr.node(a).unwrap().has_state); // nothing was evicted for it
        let exact = Tree::new(need, s).make_room(need, &[]);
        assert_eq!(exact, Ok(vec![]));
    }

    #[test]
    fn a_pool_of_zero_takes_nothing() {
        let mut tr = Tree::new(0, Sizes::of(&Dims::QWEN36));
        assert_eq!(tr.make_room(1, &[]), Err(Skip::ReuseOff));
        assert_eq!(tr.make_room(0, &[]), Err(Skip::ReuseOff));
    }

    #[test]
    fn a_later_user_query_supersedes_the_earlier_one() {
        let (mut tr, mut live, mut c) = (big(), Vec::new(), Conv::new(0, true));
        c.query();
        let (_, cut1) = c.prompt();
        let (p1, _) = c.prompt();
        c.turn(&mut tr, &mut live, false);
        let re1 = tr
            .ids()
            .into_iter()
            .find(|&i| tr.node(i).unwrap().marks.reply_end)
            .unwrap();
        c.query();
        let (p2, cut2) = c.prompt();
        c.turn(&mut tr, &mut live, false);
        let uq1 = find(&tr, &p1[..cut1.at as usize]).unwrap();
        let uq2 = find(&tr, &p2[..cut2.at as usize]).unwrap();
        let re2 = find(&tr, &live).unwrap();
        let (prot, exempt) = tr.protected(&live);
        assert_eq!(prot, BTreeSet::from([uq2, re2]));
        assert_eq!(exempt, prot);
        assert!(!prot.contains(&uq1) && !prot.contains(&re1));
    }

    #[test]
    fn the_running_requests_head_and_reply_end_are_never_evicted() {
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(s.state * 2 + s.segment(0, 150), s);
        let h = tr.insert(&t(0..100), None, Marks::USER_QUERY);
        let r = tr.insert(&t(0..150), Some(h), Marks::REPLY_END);
        let running = t(0..150);
        let (prot, exempt) = tr.protected(&running);
        assert_eq!((prot.clone(), exempt), (BTreeSet::from([h, r]), prot));
        assert_eq!(tr.used(), tr.limit());
        assert_eq!(tr.make_room(s.state, &running), Err(Skip::NoRoom));
        assert!(tr.node(h).unwrap().has_state && tr.node(r).unwrap().has_state);
        assert_eq!(tr.stats().evicted_protected, 0);
    }

    #[test]
    fn a_failed_make_room_leaves_the_tree_as_it_was() {
        let s = Sizes::of(&Dims::QWEN36);
        let limit = s.state * 2 + s.segment(0, 150) + s.segment(150, 200);
        let mut tr = Tree::new(limit, s);
        let h = tr.insert(&t(0..100), None, Marks::USER_QUERY);
        let r = tr.insert(&t(0..150), Some(h), Marks::REPLY_END);
        let x = tr.insert_parked(&t(0..200), Some(r), Marks::INTERVAL, false, 1, 1); // stateless, childless
        let used = tr.used();
        assert_eq!(used, limit);
        // sweeping x is not enough, and the running pair cannot go
        let err = tr.make_room(s.state + s.segment(150, 200) + 1, &t(0..150));
        assert_eq!(err, Err(Skip::NoRoom));
        assert_eq!((tr.len(), tr.used()), (3, used));
        assert_eq!(tr.node(x).unwrap().parent, Some(r));
        assert_eq!(tr.node(r).unwrap().children, 1);
        assert!(tr.node(h).unwrap().has_state && tr.node(r).unwrap().has_state);
        assert_chains_whole(&tr);
    }

    #[test]
    fn a_new_snapshots_parent_is_never_freed_by_its_own_eviction() {
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(s.state + s.segment(0, 5_000), s);
        let p = tr.insert(&t(0..4096), None, Marks::INTERVAL); // unprotected, childless
        let new = t(0..5_000);
        let plan = tr.plan(&new);
        let need = tr.need(plan, 5_000);
        assert_eq!(tr.make_room(need, &new), Ok(vec![Freed::State(p)]));
        assert!(!tr.node(p).unwrap().has_state);
        let Plan::New { parent, .. } = plan else {
            panic!()
        };
        let n = tr.insert(&new, parent, Marks::PROMPT_END);
        assert_eq!(tr.chain(n), vec![(p, 0, 4096), (n, 4096, 5_000)]);
        assert_eq!(tr.used(), s.state + s.segment(0, 5_000));
        assert_chains_whole(&tr);
    }

    #[test]
    fn unpark_order_brings_the_newest_back_with_its_ancestors_first() {
        let nodes = [(None, 1, 10), (Some(0), 5, 10), (None, 3, 10)];
        assert_eq!(unpark_order(&nodes, 20), vec![0, 1]);
        assert_eq!(unpark_order(&nodes, 30), vec![0, 1, 2]);
        assert_eq!(unpark_order(&nodes, 10), vec![2]);
        assert!(unpark_order(&nodes, 9).is_empty());
        assert!(unpark_order(&[], 100).is_empty());
    }

    /// The tokens behind a set of ids.
    fn named(tr: &Tree, ids: &BTreeSet<SnapId>) -> BTreeSet<Vec<u32>> {
        ids.iter()
            .map(|&i| tr.node(i).unwrap().tokens.clone())
            .collect()
    }

    /// Copies every node of `orig` into a fresh tree the way unparking does.
    fn unparked(orig: &Tree) -> Tree {
        let ids = orig.ids();
        let index: BTreeMap<SnapId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let sizes = orig.sizes();
        let input: Vec<(Option<usize>, u64, u64)> = ids
            .iter()
            .map(|&id| {
                let n = orig.node(id).unwrap();
                let from = n.parent.map_or(0, |p| orig.node(p).unwrap().pos());
                let own = sizes.segment(from, n.pos()) + if n.has_state { sizes.state } else { 0 };
                (n.parent.map(|p| index[&p]), n.last_used, own)
            })
            .collect();
        let order = unpark_order(&input, orig.used());
        assert_eq!(order.len(), ids.len());
        let mut fresh = Tree::new(orig.limit(), sizes);
        let mut new_id: BTreeMap<usize, SnapId> = BTreeMap::new();
        for i in order {
            let n = orig.node(ids[i]).unwrap();
            let parent = n.parent.map(|p| new_id[&index[&p]]);
            let id = fresh.insert_parked(
                &n.tokens,
                parent,
                n.marks,
                n.has_state,
                n.taken,
                n.last_used,
            );
            new_id.insert(i, id);
        }
        assert_eq!(fresh.used(), orig.used());
        fresh
    }

    #[test]
    fn parking_and_unparking_keep_the_lru_and_taken_order() {
        let mut tr = big();
        let mut live = Vec::new();
        let (mut a, mut b) = (Conv::new(0, true), Conv::new(1, false));
        a.query();
        b.query();
        let (pb, cutb) = b.prompt();
        for tool in [true, true, false] {
            a.turn(&mut tr, &mut live, tool);
        }
        for tool in [true, false] {
            b.turn(&mut tr, &mut live, tool);
        }
        // a long side call: interval snapshots, unprotected
        let mut side = t(600_000..609_000);
        side.extend_from_slice(&SUFFIX);
        let cut = Cut {
            at: 9_000,
            user_query: true,
        };
        serve(&mut tr, &mut live, &side, cut, &t(620_000..620_010));
        a.query();
        a.turn(&mut tr, &mut live, false);
        // a node with children loses its state
        let b_uq = find(&tr, &pb[..cutb.at as usize]).unwrap();
        assert!(tr.node(b_uq).unwrap().children > 0);
        tr.drop_state(b_uq, &live);
        // a stateless childless leaf off the running tokens: the sweep's work
        let top = tr
            .ids()
            .into_iter()
            .map(|i| tr.node(i).unwrap().last_used)
            .max()
            .unwrap();
        tr.insert_parked(
            &t(950_000..950_010),
            None,
            Marks::INTERVAL,
            false,
            top + 1,
            top + 1,
        );
        let running = live.clone();
        let s = tr.sizes();
        // more than every unprotected state can give: the protected, non-exempt tier is reached too
        let (protected_now, _) = tr.protected(&running);
        let unprotected_states = tr
            .ids()
            .into_iter()
            .filter(|i| tr.node(*i).unwrap().has_state && !protected_now.contains(i))
            .count() as u64;
        let need = tr.limit() - tr.used() + (unprotected_states + 2) * s.state;

        let mut copy = unparked(&tr);
        let tokens: BTreeMap<SnapId, Vec<u32>> = tr
            .ids()
            .into_iter()
            .map(|i| (i, tr.node(i).unwrap().tokens.clone()))
            .collect();
        let copy_tokens: BTreeMap<SnapId, Vec<u32>> = copy
            .ids()
            .into_iter()
            .map(|i| (i, copy.node(i).unwrap().tokens.clone()))
            .collect();
        let (p1, e1) = tr.protected(&running);
        let (p2, e2) = copy.protected(&running);
        assert!(!p1.is_empty() && !e1.is_empty());
        assert_eq!(named(&tr, &p1), named(&copy, &p2));
        assert_eq!(named(&tr, &e1), named(&copy, &e2));
        let key = |m: &BTreeMap<SnapId, Vec<u32>>, f: Freed| match f {
            Freed::State(i) => (true, m[&i].clone()),
            Freed::Node(i) => (false, m[&i].clone()),
        };
        let f1: Vec<_> = tr
            .make_room(need, &running)
            .unwrap()
            .into_iter()
            .map(|f| key(&tokens, f))
            .collect();
        let f2: Vec<_> = copy
            .make_room(need, &running)
            .unwrap()
            .into_iter()
            .map(|f| key(&copy_tokens, f))
            .collect();
        assert!(!f1.is_empty() && !f1[0].0, "the sweep goes first: {f1:?}");
        assert!(tr.stats().evicted_unprotected > 0 && tr.stats().evicted_protected > 0);
        assert_eq!(
            (tr.stats().evicted_unprotected, tr.stats().evicted_protected),
            (
                copy.stats().evicted_unprotected,
                copy.stats().evicted_protected
            )
        );
        assert_eq!(f1, f2);
        assert_eq!(tr.used(), copy.used());
        // a node inserted afterwards is newer than every unparked one
        let newest = copy
            .ids()
            .into_iter()
            .map(|i| {
                copy.node(i)
                    .unwrap()
                    .taken
                    .max(copy.node(i).unwrap().last_used)
            })
            .max()
            .unwrap();
        let n = copy.insert(&t(900_000..900_010), None, Marks::INTERVAL);
        assert!(copy.node(n).unwrap().taken > newest && copy.node(n).unwrap().last_used > newest);
    }

    #[test]
    fn drop_state_on_a_stateless_node_still_runs_the_cascade() {
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = big();
        let a = tr.insert(&t(0..100), None, Marks::INTERVAL);
        let b = tr.insert_parked(&t(0..300), Some(a), Marks::INTERVAL, false, 20, 20);
        let before = tr.used();
        // on the running tokens it stays; a stateless node is not "demoted" again
        assert_eq!(tr.drop_state(b, &t(0..350)), vec![]);
        assert_eq!(tr.used(), before);
        assert_eq!(tr.drop_state(b, &[]), vec![Freed::Node(b)]);
        assert_eq!(tr.used(), before - s.segment(100, 300));
        assert_eq!(tr.node(a).unwrap().children, 0);
        // a stateless node with a child stays
        let c = tr.insert_parked(&t(0..150), Some(a), Marks::INTERVAL, false, 30, 30);
        tr.insert(&t(0..200), Some(c), Marks::INTERVAL);
        assert_eq!(tr.drop_state(c, &[]), vec![]);
        assert!(tr.node(c).is_some());
    }

    #[test]
    fn the_counters_count() {
        let s = Sizes::of(&Dims::QWEN36);
        let mut tr = Tree::new(1 << 30, s);
        let a = tr.insert(&t(0..10), None, Marks::INTERVAL);
        tr.insert_parked(&t(0..20), Some(a), Marks::INTERVAL, false, 5, 5);
        tr.count_skip();
        tr.count_skip();
        tr.count_restore(100);
        tr.count_restore(23);
        let st = tr.stats();
        assert_eq!(
            (st.snapshots_held, st.pool_bytes_used, st.pool_bytes_limit),
            (1, tr.used(), 1 << 30)
        );
        assert_eq!((st.restores, st.restore_bytes, st.skipped), (2, 123, 2));
        assert_eq!(
            (
                st.reused_tokens,
                st.prefilled_tokens,
                st.pool_cut,
                st.chunks_refused
            ),
            (0, 0, false, 0)
        );
    }

    #[test]
    #[should_panic(expected = "parent")]
    fn a_parent_that_is_not_in_the_tree_is_refused() {
        let mut tr = big();
        tr.insert_parked(&t(0..100), Some(SnapId(99)), Marks::INTERVAL, true, 1, 1);
    }

    #[test]
    fn split_points_of_nothing_to_prefill_are_empty() {
        let cut = Cut {
            at: 50,
            user_query: true,
        };
        assert!(split_points(100, 100, None).is_empty());
        assert!(split_points(5_000, 100, Some(cut)).is_empty());
        assert!(split_points(0, 0, Some(cut)).is_empty());
    }
}
