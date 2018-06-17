Grouplib
A python library that analyze multiplication tables and determine corresponding Group and its properties in according to Group Theory.

What is a Group?
Let consider group $(G,)$ with operation $$ and {a, b, c} are elements in the group; then the must meet the following requirements:-

Closure: For all elements in G the result of binary operation (a ✕ b) is also in group element set.
Associativity: For all a, b and c in G, $(a * b) * c = a *(b * c)$.
Identity element: There exists an element e in G such that, for every element a in G, the equation $e * a = a * e = a$
Inverse element: For each a in G, there exists an element b in G, commonly denoted $a^{-1}$ , such that $a * b = b * a = e$ where e is the identity element.
More on Mathamatical defenination of a Group go to https://en.wikipedia.org/wiki/Group_(mathematics)

Defining a Group using Grouplib
Groups can be defined using the below methods (see Demo.ipynb for more details):

Create a group with defined operation and element set:
Grouplib.from_definition(cls, operation: callable, element_set: iter, parse: callable=str, name: str)
Passing the multiplication table via pandas.DataFrame object:
Grouplib.Group(cayley_table: pd.DataFrame, name: str)
Passing a multiplication table from a file:
Grouplib.from_file(filename: str, name: str )
Create Cyclic groups: Cyclic groups also known as Z mod n the integer modular addition mod n denoted as $(\mathbb{Z}_n, \oplus)$
Grouplib.cyclic(order:int)
Create Symmetric Group: Create group of all the permutations of nelements known as Symmetric Group denoted as $(\mathbb{S}_n, )$ where the operation $$ is permutation composition.
Group.symmetric(n: int)
Create Dihedral Groups:
Grouplib.dihedral(order:int)
Direct and Semi-Direct products of 2 groups: Let Group G & H are groups (pre-defined python group objects) and element mapping $\phi$ such that $\phi:G\rightarrow H$ Then the product can be defined as follows:
direct_product(cls,
	group_g: Group_Type,
	group_h: Group_Type)

# phi is the mapping function from G elements to H elements 
semidirect_product(cls,
	group_g: Group_Type,
	group_h: Group_Type,
	phi: Mapping_Type)
Get Group Properties
Group order, Abelian, Simple, and Solvable
Group order Group order is the number element set enclosed in the Group also known as Cardinal of the group

# Group 'g' is pre-defined
print(g.order)
Abelian Group Abelian Group is a group where all the element are commutative under its operation. $$ ab = ba ;\space \forall {a, b} \in (G,*) $$

# Group 'g' is pre-defined
print(g.is_abeailan)
Simple Group Simple Group is a Group that does not have any subgroups other than itself and the trivial Group. $$G<G \space or \space \mathbb{1}<G$$ Note: trivial group is a group of single element ${e}$

# Group 'g' is pre-defined
print(g.is_simple)
Solvable Group

# Group 'g' is pre-defined
print(g.is_solvable)
Conjugacy Classes
# Group 'g' is pre-defined
# get All Conjugacy Classes
g.conjugacy_classes
# get conjugacy class the contain element a
g.conjugacy_class_of(a: int)
Group Center is the the set of elements that are commutative to all group elements.

# Group 'g' is pre-defined
g.center
Subgroups
# Group 'g' is pre-defined
g.Subgroups
Elements Orbit
# Group 'g' is pre-defined
# Elements Orbit of the element a
g.orbit(a : int)
# All element orbits
g.orbits
# Element order
g.element_order(a : int)
# All element orders
g.elements_order
Element-wise operations
# Group 'g' is pre-defined
# Group operation
g.multiply(a: int, b: int)

#Element inverse
g.inverse(a: int)