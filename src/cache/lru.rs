use std::collections::VecDeque;
use std::hash::Hash;
use std::rc::Rc;

pub struct LruCache<K> {
    order: VecDeque<Rc<K>>,
    capacity: usize,
}

impl<K: Hash + Eq + Clone> LruCache<K> {
    pub fn new(capacity: usize) -> Self {
        Self {
            order: VecDeque::new(),
            capacity,
        }
    }

    pub fn access(&mut self, key: Rc<K>) {
        if let Some(pos) = self.order.iter().position(|k| k == &key) {
            self.order.remove(pos);
        } else if self.order.len() == self.capacity {
            self.order.pop_front();
        }
        self.order.push_back(key);
    }

    #[allow(dead_code)]
    fn evict(&mut self) -> Option<Rc<K>> {
        self.order.pop_front()
    }

    pub fn remove(&mut self, key: &Rc<K>) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
    }
}
