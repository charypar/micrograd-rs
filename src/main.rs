pub mod nn;
pub mod value;

use rand::prelude::*;

use nn::Mlp;
use value::Value;

fn main() {
    let mut rng = thread_rng();

    let xs = [
        [
            Value::new(2.0, "x_1"),
            Value::new(3.0, "x_2"),
            Value::new(-1.0, "x_3"),
        ],
        [
            Value::new(3.0, "x_1"),
            Value::new(-1.0, "x_2"),
            Value::new(0.5, "x_3"),
        ],
        [
            Value::new(0.5, "x_1"),
            Value::new(1.0, "x_2"),
            Value::new(1.0, "x_3"),
        ],
        [
            Value::new(1.0, "x_1"),
            Value::new(1.0, "x_2"),
            Value::new(-1.0, "x_3"),
        ],
    ];
    let ys = [
        Value::new(1.0, "y_1"),
        Value::new(-1.0, "y_2"),
        Value::new(-1.0, "y_3"),
        Value::new(1.0, "y_4"),
    ];

    let mlp = Mlp::new(3, &[4, 4, 1], &mut rng);

    for i in 0..500 {
        let ypred: Vec<_> = xs
            .iter()
            .map(|x| mlp.predict(x).expect("should predict")[0].clone())
            .collect();

        let loss = loss(&ys, &ypred);

        loss.backpropagate();
        mlp.nudge_parameters(0.05);

        let ypred_raw: Vec<_> = ypred.iter().map(|yp| yp.value()).collect();

        println!(
            "Iteration {}, loss {}, prediction: {:?}",
            i,
            loss.value(),
            ypred_raw
        );
    }
}

fn loss(ys: &[Value], ypred: &[Value]) -> Value {
    let loss: Value = ypred
        .iter()
        .zip(ys)
        .fold(Value::new(0.0, "0"), |sum, (ypred, y)| {
            sum + (ypred.clone() - y.clone()).pow(2.0)
        });

    loss
}
