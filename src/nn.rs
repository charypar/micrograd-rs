use rand::Rng;
use thiserror::Error as ThisError;

use crate::value::Value;

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Dimension mismatch, expected {0} inputs, got {1}")]
    DimensionMismatch(usize, usize),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Neuron {
    fn new<R: Rng>(inputs: usize, rng: &mut R) -> Self {
        let weights = (0..inputs)
            .map(|i| {
                let label = format!("w_{i}");

                Value::new(rng.gen_range(-1.0..=1.0), &label)
            })
            .collect::<Vec<_>>();

        let bias = Value::new(rng.gen_range(-1.0..=1.0), "b");

        Self { weights, bias }
    }

    fn call(&self, x: &[Value]) -> Result<Value> {
        if x.len() != self.weights.len() {
            return Err(Error::DimensionMismatch(self.weights.len(), x.len()));
        }

        Ok(self
            .weights
            .iter()
            .zip(x)
            .fold(self.bias.clone(), |sum, (w, x)| sum + w.clone() * x.clone())
            .tanh())
    }
}

#[derive(Debug)]
struct Layer {
    inputs: usize,
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new<R: Rng>(inputs: usize, outputs: usize, rng: &mut R) -> Self {
        let neurons = (0..outputs)
            .map(|_| Neuron::new(inputs, rng))
            .collect::<Vec<_>>();

        Self { inputs, neurons }
    }

    fn call(&self, x: &[Value]) -> Result<Vec<Value>> {
        if x.len() != self.inputs {
            return Err(Error::DimensionMismatch(self.inputs, x.len()));
        }

        self.neurons
            .iter()
            .map(|neuron| neuron.call(x))
            .collect::<Result<Vec<_>>>()
    }
}

pub struct Mlp {
    inputs: usize,
    layers: Vec<Layer>,
}

impl Mlp {
    pub fn new<R: Rng>(inputs: usize, layer_sizes: &[usize], rng: &mut R) -> Self {
        let layers = [&[inputs], layer_sizes]
            .concat()
            .windows(2)
            .map(|w| Layer::new(w[0], w[1], rng))
            .collect();

        Self { inputs, layers }
    }

    pub fn predict(&self, x: &[Value]) -> Result<Vec<Value>> {
        if x.len() != self.inputs {
            return Err(Error::DimensionMismatch(self.inputs, x.len()));
        }

        let init = self.layers[0].call(x)?;

        self.layers[1..]
            .iter()
            .try_fold(init, |result, layer| layer.call(&result))
    }

    pub fn nudge_parameters(&self, rate: f64) {
        for layer in &self.layers {
            for neuron in &layer.neurons {
                neuron.bias.nudge(rate);

                for weight in &neuron.weights {
                    weight.nudge(rate);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mlp, Neuron};
    use crate::value::Value;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn neuron() {
        let neuron = Neuron {
            weights: vec![
                Value::new(1.0, "w_1"),
                Value::new(0.5, "w_2"),
                Value::new(0.1, "w_3"),
            ],
            bias: Value::new(-0.3, "b"),
        };

        let out = neuron
            .call(&[
                Value::new(1.0, "x_1"),
                Value::new(2.0, "x_2"),
                Value::new(3.0, "x_3"),
            ])
            .expect("neuron call should succeed");

        assert_eq!(out.value(), 0.9640275800758169)
    }

    #[test]
    fn perceptron() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);

        let mlp = Mlp::new(3, &[4, 4, 1], &mut rng);
        let x = [
            Value::new(2.0, "x_1"),
            Value::new(3.0, "x_2"),
            Value::new(-1.0, "x_3"),
        ];

        let out = mlp.predict(&x).expect("should calculate");

        assert_eq!(out[0].value(), -0.5146818780021741);
    }
}
