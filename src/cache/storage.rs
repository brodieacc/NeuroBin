use ndarray::{ArrayD, ArrayViewD, SliceInfoElem};
use num_traits::Zero;

// Internal Storage struct for NeuroBin
//
// `Storage` struct is used internally to store and manage multi-dimensional data
// efficiently. It is not part of the public API and is meant for internal use
// within the NeuroBin library.
pub(crate) struct Storage<T> {
    data: ArrayD<T>,
}

// Implementation details of the `Storage` struct.
impl<T: Copy + Zero> Storage<T> {
    // Creates a new `Storage` with the specified shape.
    //
    // This function initializes the storage with a given shape, where each element
    // in the shape array represents the size of a dimension.
    //
    // # Arguments
    //
    // * `shape` - A slice of usize, representing the dimensions of the storage.
    pub(crate) fn new(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(shape.to_owned()),
        }
    }

    // Sets a sub-array within the `Storage`.
    //
    // This function allows setting a specific part of the storage using given
    // indices and values. It is used for internal data manipulation within NeuroBin.
    //
    // # Arguments
    //
    // * `indices` - A slice of `SliceInfoElem`, specifying the slicing information
    //               for each dimension.
    // * `values`  - A multi-dimensional array of the same type as `Storage`,
    //               representing the values to be set.
    pub(crate) fn set_subarray(&mut self, indices: &[SliceInfoElem], values: &ArrayD<T>) {
        let mut slice = self.data.slice_mut(indices);
        slice.assign(values);
    }

    // Retrieves a view of the entire data stored in `Storage`.
    //
    // This function provides read-only access to the underlying data within
    // `Storage`. It is primarily used for internal operations within NeuroBin,
    // enabling efficient data retrieval without modifications.
    //
    // # Returns
    //
    // * `ArrayViewD<T>` - A view of the entire multi-dimensional data array,
    //                     allowing read-only access to its elements.
    pub(crate) fn get_data(&self) -> ArrayViewD<T> {
        self.data.view()
    }

    // Retrieves a sub-array from `Storage`.
    //
    // This function provides a view of a specific part of the storage, as defined
    // by the provided indices.
    //
    // # Arguments
    //
    // * `indices` - A slice of `SliceInfoElem`, specifying the slicing information
    //               for each dimension.
    //
    // # Returns
    //
    // A view (`ArrayViewD`) of the sub-array.
    #[allow(dead_code)]
    pub(crate) fn get_subarray(&self, indices: &[SliceInfoElem]) -> ArrayViewD<T> {
        self.data.slice(indices)
    }
}

// Unit tests for the `Storage` struct.
//
// These tests ensure the internal functionality of `Storage` works as expected.
// They are not part of the public API tests but are crucial for internal validation.
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Test for `new` function.
    #[test]
    fn test_new() {
        let storage: Storage<i32> = Storage::new(&[2, 2]);
        assert_eq!(storage.data.shape(), &[2, 2]);
    }

    /// Test for `set_subarray` function.
    #[test]
    fn test_set_subarray() {
        let mut storage: Storage<i32> = Storage::new(&[2, 2]);
        let values = arr2(&[[1, 2], [3, 4]]);
        let indices = vec![
            SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1
            };
            values.ndim()
        ];
        let values_dyn = values.clone().into_dyn();
        storage.set_subarray(&indices, &values_dyn);
        assert_eq!(storage.data, values_dyn);
    }

    /// Test for `get_data` function.
    #[test]
    fn test_get_data() {
        let mut storage: Storage<i32> = Storage::new(&[2, 2]);
        let values = arr2(&[[1, 2], [3, 4]]);
        let indices = vec![
            SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1
            };
            values.ndim()
        ];
        let values_dyn = values.clone().into_dyn();
        storage.set_subarray(&indices, &values_dyn);
        let retrieved_data = storage.get_data();
        assert_eq!(retrieved_data, values_dyn.view());
    }
    /// Test for `get_subarray` function.
    #[test]
    fn test_get_subarray() {
        let mut storage: Storage<i32> = Storage::new(&[2, 2]);
        let values = arr2(&[[1, 2], [3, 4]]);
        let indices = vec![
            SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1
            };
            values.ndim()
        ];
        let values_dyn = values.clone().into_dyn();
        storage.set_subarray(&indices, &values_dyn);
        let retrieved_values = storage.get_subarray(&indices);
        assert_eq!(retrieved_values, values_dyn.view());
    }
}
