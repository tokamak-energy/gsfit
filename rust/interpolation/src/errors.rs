// use std::fmt;

#[derive(Debug)]
pub enum Error {
    FunctionAndXLengthMismatch { f_len: usize, x_len: usize },
    XOutOfBounds { x_desired: f64, x_min: f64, x_max: f64 },
    DuplicateXValues { x_value: f64, index: usize },
    XNotIncreasing { x_value1: f64, x_value2: f64, index: usize },
}

// impl fmt::Display for Error {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             Error::FunctionAndXLengthMismatch { f_len, x_len } => {
//                 write!(f, "function length {} != x length {}", f_len, x_len)
//             }
//             Error::XOutOfBounds { x_desired, x_min, x_max } => {
//                 write!(f, "x={} not in [{}, {}]", x_desired, x_min, x_max)
//             }
//             Error::DuplicateXValues { x_value, index } => {
//                 write!(f, "duplicate x value {} at index {}", x_value, index)
//             }
//             Error::XNotIncreasing { x_value1, x_value2, index } => {
//                 write!(f, "x not increasing at index {}: {} >= {}", index, x_value1, x_value2)
//             }
//         }
//     }
// }
