mod value;

use value::Value;

fn main() {
    let x1 = Value::new(2.0, "x1");
    let x2 = Value::new(0.0, "x2");

    let w1 = Value::new(-3.0, "w1");
    let w2 = Value::new(1.0, "w2");

    let b = Value::new(6.88137358, "b");

    let x1w1 = x1.clone() * w1;
    let x2w2 = x2 * w2;

    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b.clone();

    let o = n.clone().tanh();

    println!("n: {} = {}, grad: {}", n.label(), n.value(), n.gradient());
    println!("o: {} = {}, grad: {}", o.label(), o.value(), o.gradient());

    o.backpropagate();

    println!("n: {} = {}, grad: {}", n.label(), n.value(), n.gradient());
    println!("o: {} = {}, grad: {}", o.label(), o.value(), o.gradient());

    println!("b: {} = {}, grad: {}", b.label(), b.value(), b.gradient());
    println!(
        "x1: {} = {}, grad: {}",
        x1.label(),
        x1.value(),
        x1.gradient()
    );
    println!("b: {} = {}, grad: {}", b.label(), b.value(), b.gradient());
}
