use std::collections::VecDeque;
use std::hash::Hash;
use std::rc::Rc;

pub(crate) struct LruCache<K> {
    order: VecDeque<Rc<K>>,
    capacity: usize,
}

impl<K: Hash + Eq + Clone> LruCache<K> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            order: VecDeque::new(),
            capacity,
        }
    }

    pub(crate) fn access(&mut self, key: Rc<K>) {
        if let Some(pos) = self.order.iter().position(|k| k == &key) {
            self.order.remove(pos);
        } else if self.order.len() == self.capacity {
            self.order.pop_front();
        }
        self.order.push_back(key);
    }

    #[allow(dead_code)]
    pub(crate) fn evict(&mut self) -> Option<Rc<K>> {
        self.order.pop_front()
    }

    pub(crate) fn remove(&mut self, key: &Rc<K>) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
    }
}
