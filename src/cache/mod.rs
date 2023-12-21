mod lru;
mod storage;

use self::lru::LruCache;
use self::storage::Storage;
use ndarray::{ArrayD, Dimension, SliceInfoElem};
use num_traits::Zero;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Cache<K, T> {
    map: HashMap<Rc<K>, Storage<T>>,
    lru: LruCache<K>,
    #[allow(dead_code)]
    capacity: usize,
}

impl<K: std::hash::Hash + Eq + Clone, T: Copy + Zero> Cache<K, T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            lru: LruCache::new(capacity),
            capacity,
        }
    }

    pub fn set(&mut self, key: K, value: ArrayD<T>) -> Result<(), &'static str> {
        let rc_key = Rc::new(key);
        let shape = value.raw_dim();
        let mut storage = Storage::new(shape.slice());

        // Create the slice information directly
        let indices: Vec<SliceInfoElem> = vec![SliceInfoElem::NewAxis; shape.ndim()];
        storage.set_subarray(&indices, &value);

        self.lru.access(rc_key.clone());
        self.map.insert(rc_key, storage);
        Ok(())
    }

    pub fn get(&mut self, key: &K) -> Result<&Storage<T>, &'static str> {
        let rc_key = Rc::new(key.clone());
        if self.map.contains_key(&rc_key) {
            self.lru.access(rc_key);
            self.map.get(key).ok_or("Key not found in cache")
        } else {
            Err("Key not found in cache")
        }
    }

    pub fn delete(&mut self, key: &K) -> Result<(), &'static str> {
        let rc_key = Rc::new(key.clone());
        if self.map.remove(&rc_key).is_some() {
            self.lru.remove(&rc_key);
            Ok(())
        } else {
            Err("Key not found in cache")
        }
    }
}
