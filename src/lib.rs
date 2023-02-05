use std::cell::RefCell;
use std::cmp::PartialEq;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Sub, Mul};
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
struct ValueData {
    data: f64,
    grad: f64,
}

#[derive(Clone)]
enum Operation {
    Addition(Value, Value),
    Subtraction(Value, Value),
    Multiplication(Value, Value),
}

impl Operation {
    fn calculate_gradients(&self, grad: f64) {
        match self {
            Operation::Addition(lhs, rhs) => {
                lhs.set_grad(lhs.grad() + grad);
                rhs.set_grad(rhs.grad() + grad);
            }
            Operation::Subtraction(lhs, rhs) =>  {
                lhs.set_grad(lhs.grad() + grad);
                rhs.set_grad(rhs.grad() - grad);
            }
            Operation::Multiplication(lhs, rhs) =>  {
                lhs.set_grad(grad * rhs.data());
                rhs.set_grad(grad * lhs.data());
            }
        }
    }
}

#[derive(Clone)]
pub struct Value {
    data: Rc<RefCell<ValueData>>,
    operation: Option<Rc<Operation>>,
}

impl Value {
    pub fn from_val(val: f64) -> Self {
        Self::new(val, None)
    }

    fn new(data: f64, operation: Option<Rc<Operation>>) -> Self {
        Self {
            data: Rc::new(RefCell::new(ValueData { data, grad: 0.0 })),
            operation,
        }
    }

    pub fn data(&self) -> f64 {
        self.data.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.data.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        let self_grad = &mut RefCell::borrow_mut(&self.data).grad;
        *self_grad = grad;
    }

    pub fn backward(&self) {
        self.set_grad(1.0);
        for value in self.toposort() {
            if let Some(operation) = &value.operation {
                operation.calculate_gradients(value.grad());
            }
        }
    }

    fn toposort(&self) -> Vec<&Value> {
        let mut ordering = vec![];
        let mut visited = HashSet::new();
        self.toposort_impl(&mut visited, &mut ordering);
        ordering.reverse();
        ordering
    }

    fn toposort_impl<'a>(
        &'a self,
        visited: &mut HashSet<&'a Value>,
        traversal: &mut Vec<&'a Value>,
    ) {
        if visited.contains(&self) {
            return;
        }
        visited.insert(self);

        match self.operation.as_ref().map(|op| op.as_ref()) {
            Some(Operation::Addition(lhs, rhs)) => {
                lhs.toposort_impl(visited, traversal);
                rhs.toposort_impl(visited, traversal);
            }
            Some(Operation::Subtraction(lhs, rhs)) => {
                lhs.toposort_impl(visited, traversal);
                rhs.toposort_impl(visited, traversal);
            }
            Some(Operation::Multiplication(lhs, rhs)) => {
                lhs.toposort_impl(visited, traversal);
                rhs.toposort_impl(visited, traversal);
            }
            None => {}
        }

        traversal.push(self);
    }
}


impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.data).hash(state);
    }
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        Value::new(
            self.data() + other.data(),
            Some(Rc::new(Operation::Addition(self, other)))
        )
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Value {
        Value::new(
            self.data() - other.data(),
            Some(Rc::new(Operation::Subtraction(self, other)))
        )
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        Value::new(
            self.data() * other.data(),
            Some(Rc::new(Operation::Multiplication(self, other)))
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_value_reuse() {
        let a = Value::from_val(1.0);
        let b = a.clone() + a.clone();
        b.backward();

        assert_eq!(a.grad(), 2.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn test_add() {
        let a = Value::from_val(1.0);
        let b = Value::from_val(2.0);
        let c = a.clone() + b.clone();
        c.backward();

        assert_eq!(c.data(), 3.0);
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn test_sub() {
        let a = Value::from_val(1.0);
        let b = Value::from_val(2.0);
        let c = a.clone() - b.clone();
        c.backward();

        assert_eq!(c.data(), -1.0);
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), -1.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn test_mul() {
        let a = Value::from_val(11.0);
        let b = Value::from_val(12.0);
        let c = a.clone() * b.clone();
        c.backward();

        assert_eq!(c.data(), 132.0);
        assert_eq!(a.grad(), 12.0);
        assert_eq!(b.grad(), 11.0);
        assert_eq!(c.grad(), 1.0);
    }
}
