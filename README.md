# Grouplib
A python library that analyze multiplication tables and determine corresponding Group and its properties in according to Group Theory.

## What is a Group?
 Let concider group G with operation * $(G,*)$ and <b>a</b>, <b>b</b> and <b>c</b> are elements in the group; Then:-
- **Closure**: 
  For all elements in G the result of binary operation (a ✕ b) is also in group element set.
- **Associativity**:
  For all a, b and c in G, $(a ✕ b) ✕ c = a ✕ (b ✕ c)$.
- **Identity element**:
  There exists an element **e** in G such that, for every element a in G, the equation 
$e ✕ a = a ✕ e = a$
- **Inverse element**:
  For each a in G, there exists an element b in G, commonly denoted $a^{-1}$ , such that 
$a ✕ b = b ✕ a = e$
where e is the identity element.

More on Mathamatical defenination of a Group go to https://en.wikipedia.org/wiki/Group_(mathematics)


## Defining a Group using Grouplib

Groups can be defined using the below methods (see Demo.ipynb for more details):
- Create a group with defined operation and element set:
```python
Grouplib.from_definition(cls, operation: callable, element_set: iter, parse: callable=str, name: str)-> Group_Type
```
- Passing the multiplication table via pandas.DataFrame object.
```python
Grouplib.Group(cayley_table: pd.DataFrame, name: str)-> Group_Type
```
- Passing a multiplication table from a file:
```python
Grouplib.from_file(filename: str, name: str )->Group_Type
```
- Create Cyclic groups using :
```python 
Grouplib.cyclic(order:int)-> Group_Type
```
- Direct and Semi-Direct products of 2 groups:
Let Group G & H are groups (pre-defined python group objects) and  element mapping <img src="http://bit.ly/2xJLjLF" align="center" border="0" alt="$\phi$ " width="17" height="19" /> such that <img src="http://bit.ly/2sycHHo" align="center" border="0" alt="$\phi:G\rightarrow H$ " width="83" height="19" />
Then the product can be defined as follows:

```python
direct_product(cls,
	group_g: Group_Type,
	group_h: Group_Type) -> Group_Type

# phi is the mapping function from G elements to H elements 
semidirect_product(cls,
	group_g: Group_Type,
	group_h: Group_Type,
	phi: Mapping_Type) -> Group_Type
```








