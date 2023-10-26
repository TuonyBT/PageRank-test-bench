extern crate nalgebra as na;

use std::ops::Mul;

use na::{Vector3, RowVector3, Matrix3, VectorView3, Matrix, DMatrix, Const, ArrayStorage, DVector};

const THRESHOLD: f32 = 0.0000000000001;
const BETA: f32 = 1.0;

fn main() {

    // adjacency matrix
    let arr = Matrix3::<f32>::from_columns(
        &[Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(1.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0)]
    );
    println!("Points awarded matrix: {:?}", arr);

    let s = arr.row_sum_tr();
    println!("Summation of columns: {:?}", s);

    let m = Matrix3::<f32>::from_columns( 
                &arr.column_iter().zip(&s).map(|(c, &cs)| c / cs)
                .collect::<Vec<Matrix<f32, Const<3>, Const<1>, ArrayStorage<f32, 3, 1>>>>()
            );
    println!("Column stochastic probability matrix M: {:?}", m);


    let r_new = Vector3::from_element(1.0 / m.ncols() as f32);
    println!("Initial rank vector: {:?}", r_new);

    let c = r_new.clone() * (1.0 - BETA);
    println!("Teleportation vector: {:?}", c);

    let mut r_prev = r_new.clone();
    println!();

    for i in 1..1001 {
        print!("Iteration {}; ", i);

        let r_new = BETA * m.mul(&r_prev) + &c;
        print!("r_new: {:?}; ", r_new);

        let diff = (r_new - &r_prev).abs().sum();
        println!("diff: {:?}", diff);

        if diff < THRESHOLD {break}

        r_prev = r_new;
    }

    println!("The final rank vector: {:?}", r_prev);

}


fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}