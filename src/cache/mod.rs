mod lru;
mod storage;

use self::lru::LruCache;
use self::storage::Storage;
use ndarray::{ArrayD, ArrayViewD, Dimension, SliceInfoElem};
use num_traits::Zero;
use std::collections::HashMap;
use std::rc::Rc;

/// The `Cache` struct represents a distributed in-memory cache optimized for LLMs.
/// It stores multi-dimensional data and uses a Least Recently Used (LRU) eviction policy.
pub struct Cache<K, T> {
    map: HashMap<Rc<K>, Storage<T>>,
    lru: LruCache<K>,
    capacity: usize,
}

impl<K: std::hash::Hash + Eq + Clone, T: Copy + Zero> Cache<K, T> {
    /// Creates a new `Cache` with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - The maximum number of items the cache can hold.
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            lru: LruCache::new(capacity),
            capacity,
        }
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// If the cache has reached its capacity, the least recently used item is evicted
    /// before inserting the new key-value pair. This ensures that the cache size
    /// remains within the defined limits.
    ///
    /// # Arguments
    /// * `key` - The key associated with the value.
    /// * `value` - The value to store in the cache.
    ///
    /// # Returns
    /// * `Ok(())` - If the insertion is successful.
    /// * `Err(&'static str)` - If the cache is full and unable to evict items, or if
    ///                         any other error occurs.
    pub fn set(&mut self, key: K, value: ArrayD<T>) -> Result<(), &'static str> {
        let rc_key = Rc::new(key);
        let shape = value.raw_dim();
        let mut storage = Storage::new(shape.slice());

        // Create the slice information directly
        let indices: Vec<SliceInfoElem> = vec![SliceInfoElem::NewAxis; shape.ndim()];
        storage.set_subarray(&indices, &value);

        if self.map.len() >= self.capacity {
            // Evict the least recently used item
            if let Some(evicted_key) = self.lru.evict() {
                self.map.remove(&evicted_key);
            } else {
                return Err("Cache is full and unable to evict items");
            }
        }

        self.lru.access(rc_key.clone());
        self.map.insert(rc_key, storage);
        Ok(())
    }

    /// Retrieves a value associated with the given key from the cache.
    ///
    /// # Arguments
    /// * `key` - The key for which to retrieve the value.
    ///
    /// # Returns
    /// * A result containing a reference to the value if found, or an error message if not.
    pub fn get(&mut self, key: &K) -> Result<ArrayViewD<T>, &'static str> {
        let rc_key = Rc::new(key.clone());
        if let Some(storage) = self.map.get(&rc_key) {
            self.lru.access(rc_key);
            Ok(storage.get_data())
        } else {
            Err("Key not found in cache")
        }
    }

    /// Deletes a key-value pair from the cache.
    ///
    /// # Arguments
    /// * `key` - The key of the pair to delete.
    ///
    /// # Returns
    /// * A result indicating whether the operation was successful.
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr0;

    #[test]
    fn test_cache_initialization() {
        let cache: Cache<i32, i32> = Cache::new(2);
        assert_eq!(cache.map.len(), 0);
        assert_eq!(cache.lru.capacity, 2);
    }

    #[test]
    fn test_cache_set_and_get() {
        let mut cache: Cache<i32, i32> = Cache::new(2);
        let value = arr0(42).into_dyn();
        assert_eq!(cache.set(1, value.clone()), Ok(()));
        assert_eq!(cache.get(&1), Ok(value.view()));
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache: Cache<i32, i32> = Cache::new(2);
        let value1 = arr0(42).into_dyn();
        let value2 = arr0(43).into_dyn();
        let value3 = arr0(44).into_dyn();
        assert_eq!(cache.set(1, value1.clone()), Ok(()));
        assert_eq!(cache.set(2, value2.clone()), Ok(()));
        assert_eq!(cache.set(3, value3.clone()), Ok(()));
        assert_eq!(cache.get(&1), Err("Key not found in cache"));
        assert_eq!(cache.get(&2), Ok(value2.view()));
        assert_eq!(cache.get(&3), Ok(value3.view()));
    }

    #[test]
    fn test_cache_delete_existing_key() {
        let mut cache: Cache<i32, i32> = Cache::new(2);
        let value = arr0(42).into_dyn();
        assert_eq!(cache.set(1, value.clone()), Ok(()));
        assert_eq!(cache.delete(&1), Ok(()));
        assert_eq!(cache.get(&1), Err("Key not found in cache"));
    }

    #[test]
    fn test_cache_delete_nonexistent_key() {
        let mut cache: Cache<i32, i32> = Cache::new(2);
        assert_eq!(cache.delete(&1), Err("Key not found in cache"));
    }
}
