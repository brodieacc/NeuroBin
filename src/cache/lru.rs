use std::collections::VecDeque;
use std::hash::Hash;
use std::rc::Rc;

/// `LruCache` is an internal structure used by NeuroBin for implementing LRU caching.
///
/// This structure manages the caching of keys using a Least Recently Used (LRU) strategy.
/// It is designed for internal use within the NeuroBin library and not exposed in the public API.
pub(crate) struct LruCache<K> {
    order: VecDeque<Rc<K>>,
    capacity: usize,
}

/// Implementation details of the `LruCache` struct.
impl<K: Hash + Eq + Clone> LruCache<K> {
    /// Creates a new LRU cache with the specified capacity.
    ///
    /// Initializes an LRU cache that can hold a maximum of `capacity` items.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The maximum number of items the cache can hold.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            order: VecDeque::new(),
            capacity,
        }
    }

    /// Accesses an item in the cache, marking it as recently used.
    ///
    /// Moves the accessed item to the end of the order queue, indicating recent use.
    /// If the item is not in the cache and the cache is full, the least recently used item
    /// is removed.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the item being accessed.
    pub(crate) fn access(&mut self, key: Rc<K>) {
        if let Some(pos) = self.order.iter().position(|k| k == &key) {
            self.order.remove(pos);
        } else if self.order.len() == self.capacity {
            self.order.pop_front();
        }
        self.order.push_back(key);
    }

    /// Evicts the least recently used item from the cache.
    ///
    /// This method removes the oldest item from the cache, typically used when
    /// the cache is full. It's a key component of the LRU mechanism.
    ///
    /// # Returns
    /// * `Option<Rc<K>>` - The key of the evicted item, if any.
    pub(crate) fn evict(&mut self) -> Option<Rc<K>> {
        self.order.pop_front()
    }

    /// Removes a specific item from the cache based on the key.
    ///
    /// If the key is found in the cache, it is removed along with its associated value.
    ///
    /// # Arguments
    ///
    /// * `key` - A reference to the key of the item to remove.
    pub(crate) fn remove(&mut self, key: &Rc<K>) -> bool {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn test_lru_cache_initialization() {
        let lru: LruCache<i32> = LruCache::new(2);
        assert_eq!(lru.order.len(), 0);
        assert_eq!(lru.capacity, 2);
    }

    #[test]
    fn test_lru_cache_access() {
        let mut lru = LruCache::new(2);
        lru.access(Rc::new(1));
        lru.access(Rc::new(2));
        lru.access(Rc::new(1));

        assert_eq!(lru.order.len(), 2);
        assert_eq!(*lru.order.front().unwrap(), Rc::new(2));
        assert_eq!(*lru.order.back().unwrap(), Rc::new(1));
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut lru = LruCache::new(2);
        lru.access(Rc::new(1));
        lru.access(Rc::new(2));
        lru.access(Rc::new(3)); // This should evict '1'

        assert_eq!(lru.order.len(), 2);
        assert!(lru.order.contains(&Rc::new(2)));
        assert!(lru.order.contains(&Rc::new(3)));
        assert!(!lru.order.contains(&Rc::new(1)));
    }

    #[test]
    fn test_lru_cache_item_removal() {
        let mut lru = LruCache::new(2);
        let key = Rc::new(1);
        lru.access(key.clone());
        lru.remove(&key);

        assert!(lru.order.is_empty());
    }

    #[test]
    fn test_lru_cache_capacity() {
        let mut lru = LruCache::new(2);
        lru.access(Rc::new(1));
        lru.access(Rc::new(2));
        lru.access(Rc::new(3));

        assert_eq!(lru.order.len(), 2);
    }

    #[test]
    fn test_lru_cache_order_maintenance() {
        let mut lru = LruCache::new(3);
        lru.access(Rc::new(1));
        lru.access(Rc::new(2));
        lru.access(Rc::new(3));
        lru.access(Rc::new(2));

        assert_eq!(*lru.order.front().unwrap(), Rc::new(1));
        assert_eq!(*lru.order.back().unwrap(), Rc::new(2));
    }

    #[test]
    fn test_lru_cache_edge_cases() {
        let mut lru = LruCache::new(2);
        let non_existing_key = Rc::new(99);

        assert!(!lru.remove(&non_existing_key));
        assert!(lru.order.is_empty());
    }
}
