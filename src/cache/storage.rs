use ndarray::{ArrayD, ArrayViewD, SliceInfoElem};
use num_traits::Zero;

pub struct Storage<T> {
    data: ArrayD<T>,
}

impl<T: Copy + Zero> Storage<T> {
    pub fn new(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(shape.to_owned()),
        }
    }

    pub fn set_subarray(&mut self, indices: &[SliceInfoElem], values: &ArrayD<T>) {
        let mut slice = self.data.slice_mut(indices);
        slice.assign(values);
    }

    pub fn get_subarray(&self, indices: &[SliceInfoElem]) -> ArrayViewD<T> {
        self.data.slice(indices)
    }
}
