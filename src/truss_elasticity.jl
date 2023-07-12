# In this tutorial, we will learn how to simulate single [truss](https://en.wikipedia.org/wiki/Truss) using Gridap
#
# ## Problem statement
#
# In this tutorial, we detail how to solve a linear truss having square cross section of
# 0.001x0.001 m (A=1e-6 $m^2$) and length of 0.2 m.
#
#
# In first phase we use [one-dimensional approximation based on Hooke's law](https://en.wikipedia.org/wiki/Hooke%27s_law)
# We impose the following boundary conditions. 
# Displacement is constrained to zero on the left end and pulling load of 100 N is applied at right end.
# 
# Formally, the PDE to solve is
#
# ```math
# \begin{aligned}
# \epsilon = \frac{F}{A E}
# u = 0 \ &\text{on}\ x = 0,\\
# f(0.2) = 100,\\
# \end{aligned}
# ```
#
# The variable $u$ stands for the unknown displacement vector $\sigma(u)$ is the stress tensor defined as
# ```math
# \sigma(u) = f / A,
# ```
# Here, we consider material parameters corresponding to steel with Young's modulus $E=210\cdot 10^9$ Pa.
# For 1D solution Poisson's ratio is not taken into account.
#
# ## Numerical scheme
#
#
# ## Discrete model
#
# We start by defining model
using Gridap

# In order to inspect it, write the model to vtk

writevtk(model,"model")

# and open the resulting files with Paraview. 
# The next step is the construction of the FE space. Here, we need to build a vector-valued FE space, which is done as follows:

order = 1

reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
V0 = TestFESpace(model,reffe;
  conformity=:H1,
  dirichlet_tags=["surface_1","surface_2"],
  dirichlet_masks=[(true,false,false), (true,true,true)])

# As in previous tutorial, we construct a continuous Lagrangian interpolation of order 1. 
# The vector-valued interpolation is selected via the option `valuetype=VectorValue{1,Float64}`, where we use the type `VectorValue{3,Float64}`, 
# which is the way Gridap represents vectors of three `Float64` components. 
# We mark as Dirichlet the objects identified with the tags `"surface_1"` and `"surface_2"` using the `dirichlet_tags` argument. 
# Finally, we chose which components of the displacement are actually constrained on the Dirichlet boundary via the `dirichlet_masks` argument. 
# Note that we constrain only the first component on the boundary $\Gamma_{\rm B}$ (identified as `"surface_1"`), 
# whereas we constrain all components on $\Gamma_{\rm G}$ (identified as `"surface_2"`).
#
# The construction of the trial space is slightly different in this case. 
# The Dirichlet boundary conditions are described with two different functions, one for boundary $\Gamma_{\rm B}$ and another one for $\Gamma_{\rm G}$. These functions can be defined as

g1(x) = VectorValue(0.005,0.0,0.0)
g2(x) = VectorValue(0.0,0.0,0.0)

# From functions `g1` and `g2`, we define the trial space as follows:

U = TrialFESpace(V0,[g1,g2])

# Note that the functions `g1` and `g2` are passed to the `TrialFESpace` constructor in the same order as the boundary identifiers are passed previously in the `dirichlet_tags` argument of the `TestFESpace` constructor.
#
# ## Constitutive law
#
# Once the FE spaces are defined, the next step is to define the weak form.  In this example, the construction of the weak form requires more work than in previous tutorial since we need to account for the constitutive law that relates strain and stress.  The symmetric gradient operator is represented by the function `ε` provided by Gridap (also available as `symmetric_gradient`). However, function `σ` representing the stress tensor is not predefined in the library and it has to be defined ad-hoc by the user, namely

const E = 70.0e9
const ν = 0.33
const λ = (E*ν)/((1+ν)*(1-2*ν))
const μ = E/(2*(1+ν))
σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

# Function `σ` takes a strain tensor `ε`(one can interpret this strain as the strain at an arbitrary integration point) and computes the associated stress tensor using the Lamé operator.  Note that the implementation of function `σ` is very close to its mathematical definition.
#
#  ## Weak form
#
#  As seen in previous tutorials, in order to define the weak form we need to build the integration mesh and the corresponding measure

degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

#  From these objects and the constitutive law previously defined, we can write the weak form as follows

a(u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ
l(v) = 0

# Note that we have composed function `σ` with the strain field `ε(u)` in order to compute the stress field associated with the trial function `u`. The linear form is simply `l(v) = 0` since there are not external forces in this example.
#
# ## Solution of the FE problem
#
# The remaining steps for solving the FE problem are essentially the same as in previous tutorial.

op = AffineFEOperator(a,l,U,V0)
uh = solve(op)

# Note that we do not have explicitly constructed a `LinearFESolver` in order to solve the FE problem. If a `LinearFESolver` is not passed to the `solve` function, a default solver (LU factorization) is created and used internally.
#
# Finally, we write the results to a file. Note that we also include the strain and stress tensors into the results file.

writevtk(Ω,"results",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ∘ε(uh)])

# It can be clearly observed (see next figure) that the surface  $\Gamma_{\rm B}$ is pulled in $x_1$-direction and that the solid deforms accordingly.
#
# ![](../assets/elasticity/disp_ux_40.png)

# ## Multi-material problems
#
# We end this tutorial by extending previous code to deal with multi-material problems. Let us assume that the piece simulated before is now made of 2 different materials (see next figure). In particular, we assume that the volume depicted in dark green is made of aluminum, whereas the volume marked in purple is made of steel.
#
# ![](../models/solid-mat.png)
#
# The two different material volumes are properly identified in the model we have previously loaded. To check this, inspect the model with Paraview (by writing it to vtk format as done before). Note that the volume made of aluminum is identified as `"material_1"`, whereas the volume made of steel is identified as `"material_2"`.
#
# In order to build the constitutive law for the bi-material problem, we need a vector that contains information about the material each cell in the model is composed. This is achieved by these lines

using Gridap.Geometry
labels = get_face_labeling(model)
dimension = 3
tags = get_face_tag(labels,dimension)

# Previous lines generate a vector, namely `tags`, whose length is the number of cells in the model and for each cell contains an integer that identifies the material of the cell.  This is almost what we need. We also need to know which is the integer value associated with each material. E.g., the integer value associated with `"material_1"` (i.e. aluminum) is retrieved as

const alu_tag = get_tag_from_name(labels,"material_1")

# Now, we know that cells whose corresponding value in the `tags` vector is `alu_tag` are made of aluminum, otherwise they are made of steel (since there are only two materials in this example).
#
# At this point, we are ready to define the multi-material constitutive law. First, we define the material parameters for aluminum and steel respectively:

function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end

const E_alu = 70.0e9
const ν_alu = 0.33
const (λ_alu,μ_alu) = lame_parameters(E_alu,ν_alu)

const E_steel = 200.0e9
const ν_steel = 0.33
const (λ_steel,μ_steel) = lame_parameters(E_steel,ν_steel)

# Then, we define the function containing the constitutive law:

function σ_bimat(ε,tag)
  if tag == alu_tag
    return λ_alu*tr(ε)*one(ε) + 2*μ_alu*ε
  else
    return λ_steel*tr(ε)*one(ε) + 2*μ_steel*ε
  end
end

# Note that in this new version of the constitutive law, we have included a third argument that represents the integer value associated with a certain material. If the value corresponds to the one for aluminum (i.e., `tag == alu_tag`), then, we use the constitutive law for this material, otherwise, we use the law for steel.
#
# Since we have constructed a new constitutive law, we need to re-define the bilinear form of the problem:

a(u,v) = ∫( ε(v) ⊙ (σ_bimat∘(ε(u),tags)) )*dΩ

# In previous line, pay attention in the usage of the new constitutive law `σ_bimat`. Note that we have passed the vector `tags` containing the material identifiers in the last argument of the function`.
#
# At this point, we can build the FE problem again and solve it

op = AffineFEOperator(a,l,U,V0)
uh = solve(op)

# Once the solution is computed, we can store the results in a file for visualization. Note that, we are including the stress tensor in the file (computed with the bi-material law).

writevtk(Ω,"results_bimat",cellfields=
  ["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ_bimat∘(ε(uh),tags)])


#  Tutorial done!
