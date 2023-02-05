use std::cell::RefCell;
use std::cmp::PartialEq;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
struct ValueData {
    data: f64,
    grad: f64,
}

#[derive(Clone)]
enum Operation {
    Addition(Rc<Value>, Rc<Value>),
}

impl Operation {
    fn calculate_gradients(&self, grad: f64) {
        match self {
            Operation::Addition(lhs, rhs) => {
                // These could both be pointing to the same value (e.g. a + a)
                // So make sure to drop the reference to self_data before updating other_grad
                {
                    let lhs_grad = &mut RefCell::borrow_mut(&lhs.data).grad;
                    *lhs_grad += grad;
                }

                let rhs_grad = &mut RefCell::borrow_mut(&rhs.data).grad;
                *rhs_grad += grad;
            }
        }
    }
}

#[derive(Clone)]
pub struct Value {
    data: Rc<RefCell<ValueData>>,
    operation: Option<Operation>,
}

impl Value {
    pub fn from_val(val: f64) -> Self {
        Self::new(val, None)
    }

    fn new(data: f64, operation: Option<Operation>) -> Self {
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
        let order = toposort(self);
        for node in order {
            if let Some(operation) = &node.operation {
                operation.calculate_gradients(node.grad());
            }
        }
    }
}

fn toposort(value: &Value) -> Vec<&Value> {
    let mut ordering = vec![];
    let mut visited = HashSet::new();
    toposort_impl(value, &mut visited, &mut ordering);
    ordering.reverse();
    ordering
}

fn toposort_impl<'a>(
    value: &'a Value,
    visited: &mut HashSet<&'a Value>,
    traversal: &mut Vec<&'a Value>,
) {
    if visited.contains(&value) {
        return;
    }
    visited.insert(value);

    match &value.operation {
        Some(Operation::Addition(lhs, rhs)) => {
            toposort_impl(lhs.as_ref(), visited, traversal);
            toposort_impl(rhs.as_ref(), visited, traversal);
        }
        None => {}
    }

    traversal.push(value);
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

// impl std::ops::Neg for Value {
//     type Output = Value;

//     fn neg(self) -> Value {
//         Value::new(-self.data(), vec![Rc::new(self)], None)
//     }
// }

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        Value::new(
            self.data() + other.data(),
            Some(Operation::Addition(Rc::new(self), Rc::new(other)))
        )
    }
}

// impl std::ops::Sub for Value {
//     type Output = Value;

//     fn sub(self, other: Value) -> Value {
//         self + (-other)
//     }
// }

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
        assert_eq!(c.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
        assert_eq!(c.grad(), 1.0);
    }
}
