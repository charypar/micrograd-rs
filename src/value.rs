use std::{
    cell::RefCell,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Debug)]
enum Operation {
    Constant,
    Add(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Sub(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Pow(Rc<RefCell<ValueInner>>, f64),
    Multiply(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Tanh(Rc<RefCell<ValueInner>>),
}

#[derive(Debug)]
struct ValueInner {
    value: f64,
    label: String,
    gradient: f64,
    operation: Operation,
}

impl ValueInner {
    fn backpropagate(&self) {
        match &self.operation {
            Operation::Constant => (),
            Operation::Add(lhs, rhs) => {
                lhs.borrow_mut().gradient += self.gradient;
                rhs.borrow_mut().gradient += self.gradient;

                lhs.borrow().backpropagate();
                rhs.borrow().backpropagate();
            }
            Operation::Sub(lhs, rhs) => {
                lhs.borrow_mut().gradient += self.gradient;
                rhs.borrow_mut().gradient -= self.gradient;

                lhs.borrow().backpropagate();
                rhs.borrow().backpropagate();
            }
            Operation::Multiply(lhs, rhs) => {
                lhs.borrow_mut().gradient += rhs.borrow().value * self.gradient;
                rhs.borrow_mut().gradient += lhs.borrow().value * self.gradient;

                lhs.borrow().backpropagate();
                rhs.borrow().backpropagate();
            }
            Operation::Pow(it, exponent) => {
                let val = it.borrow().value;

                it.borrow_mut().gradient += (exponent * val.powf(*exponent - 1.0)) * self.gradient;
                it.borrow().backpropagate()
            }
            Operation::Tanh(it) => {
                it.borrow_mut().gradient += (1.0 - self.value.powf(2.0)) * self.gradient;
                it.borrow().backpropagate();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Value {
    inner: Rc<RefCell<ValueInner>>,
}

impl Value {
    pub fn new(value: f64, label: &str) -> Self {
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                value,
                label: label.to_string(),
                gradient: 0.0,
                operation: Operation::Constant,
            })),
        }
    }

    pub fn value(&self) -> f64 {
        self.inner.borrow().value
    }

    pub fn gradient(&self) -> f64 {
        self.inner.borrow().gradient
    }

    pub fn label(&self) -> String {
        self.inner.borrow().label.clone()
    }

    pub fn tanh(self) -> Value {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                value: self.inner.borrow().value.tanh(),
                label: format!("tanh({})", self.inner.borrow().label),
                gradient: 0.0,
                operation: Operation::Tanh(self.inner.clone()),
            })),
        }
    }

    pub fn pow(self, exponent: f64) -> Value {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                value: self.inner.borrow().value.powf(exponent),
                label: format!("{}^{}", self.inner.borrow().label, exponent),
                gradient: 0.0,
                operation: Operation::Pow(self.inner.clone(), exponent),
            })),
        }
    }

    pub fn backpropagate(&self) {
        // Kick off with a gradient of 1
        self.inner.borrow_mut().gradient = 1.0;

        // propagate through the graph
        self.inner.borrow().backpropagate()
    }

    pub fn nudge(&self, rate: f64) {
        let grad = self.inner.borrow().gradient;
        let mut inner = self.inner.borrow_mut();

        inner.value -= rate * grad;
        inner.gradient = 0.0; // reset gradient to avoid accumulating in subsequent backpropagations
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                value: self.inner.borrow().value * rhs.inner.borrow().value,
                label: format!(
                    "({} * {})",
                    self.inner.borrow().label,
                    rhs.inner.borrow().label
                ),
                gradient: 0.0,
                operation: Operation::Multiply(self.inner.clone(), rhs.inner.clone()),
            })),
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                value: self.inner.borrow().value + rhs.inner.borrow().value,
                label: format!(
                    "({} + {})",
                    self.inner.borrow().label,
                    rhs.inner.borrow().label
                ),
                gradient: 0.0,
                operation: Operation::Add(self.inner.clone(), rhs.inner.clone()),
            })),
        }
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        Value {
            inner: Rc::new(RefCell::new(ValueInner {
                value: self.inner.borrow().value - rhs.inner.borrow().value,
                label: format!(
                    "({} - {})",
                    self.inner.borrow().label,
                    rhs.inner.borrow().label
                ),
                gradient: 0.0,
                operation: Operation::Sub(self.inner.clone(), rhs.inner.clone()),
            })),
        }
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * Value::new(-1.0, "-1")
    }
}

#[cfg(test)]
mod tests {
    use super::Value;

    #[test]
    fn constant() {
        let c = Value::new(5.0, "c");

        assert_eq!(5.0, c.value());
        assert_eq!("c", c.label());
    }

    #[test]
    fn add() {
        let a = Value::new(3.0, "a");
        let b = Value::new(5.0, "b");
        let c = a + b;

        assert_eq!(8.0, c.value());
        assert_eq!("(a + b)", c.label());
    }

    #[test]
    fn multiply() {
        let a = Value::new(3.0, "a");
        let b = Value::new(5.0, "b");
        let c = a * b;

        assert_eq!(15.0, c.value());
        assert_eq!("(a * b)", c.label());
    }

    #[test]
    fn expression() {
        let x1 = Value::new(2.0, "x1");
        let x2 = Value::new(0.0, "x2");

        let w1 = Value::new(-3.0, "w1");
        let w2 = Value::new(1.0, "w2");

        let b = Value::new(6.88137358, "b");

        let x1w1 = x1 * w1;
        let x2w2 = x2 * w2;

        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;

        let o = n.tanh();

        assert_eq!(o.value(), 0.707106777676776);
    }

    #[test]
    fn backpropagation() {
        let a = Value::new(-2.0, "a");
        let b = Value::new(3.0, "b");

        let d = a.clone() * b.clone();
        let e = a.clone() + b.clone();
        let f = d.clone() * e.clone();

        f.backpropagate();

        assert_eq!(f.gradient(), 1.0);
        assert_eq!(e.gradient(), -6.0);
        assert_eq!(d.gradient(), 1.0);
        assert_eq!(a.gradient(), -3.0);
        assert_eq!(b.gradient(), -8.0);
    }

    #[test]
    fn backpropagation_pow() {
        let a = Value::new(3.0, "x");
        let b = a.clone().pow(2.0);

        b.backpropagate();

        assert_eq!(a.gradient(), 6.0)
    }

    #[test]
    fn backpropagation_sub() {
        let a = Value::new(3.0, "a");
        let b = Value::new(2.0, "b");
        let c = a.clone() - b.clone();

        let d = Value::new(3.0, "d");
        let e = Value::new(2.0, "e");
        let f = d.clone() + (e.clone() * Value::new(-1.0, "-1"));

        c.backpropagate();
        f.backpropagate();

        assert_eq!(a.gradient(), d.gradient());
        assert_eq!(b.gradient(), e.gradient());
    }
}
