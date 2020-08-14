
//Including all "includes" from step-46 to debug compilation issues

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <iostream>
#include <fstream>

/*
TODO:
* Include constants in class template
* Replace London penetration depth with multi-domain implementation
*/


namespace VectorPoissonSolver
{
  using namespace dealii;

  template <int dim>
  class PoissonSolver
  {
  public:
    PoissonSolver(const unsigned int london_degree,
		    const unsigned int maxwell_degree);

    // The method called by "main" to kick off the computation.
    void run();

  private:
    enum 
    {
	sample_id,
	vacuum_id
    };

    static bool is_in_sample(
	const typename hp::DoFHandler::cell_iterator &cell);

    // Initializes containers, distributes DOFs, and calculates constraints.
    void setup_system();

    // Calculates the system matrix and RHS from the weak form of the problem.
    void assemble_system();

    // Solves the linear system.
    void solve();

    // Estimates errors and marks cells for refinement.
    void refine_grid();

    // Writes results to files for visualization.
    void output_results(const unsigned int cycle) const;

    // Container for the mesh (discretization).
    Triangulation<dim> triangulation;


    /*
     *'london_fe' --- system inside the sample
     *'maxwell_fe' --- system outside the sample
     */
    FESystem<dim> london_fe;
    FESystem<dim> maxwell_fe;
    const unsigned int london_degree;
    const unsigned int maxwell_degree;

/*    We need to use collections from the 'hp' namespace because the elements may be region-specific.*/
    hp::FECollection<dim> fe_collection;
    hp::DoFHandler<dim> dof_handler;

    // This is the container which will end up holding all the constraints in the problem
    // (i.e., boundary conditions and "unphysical" results of uneven mesh refinement called hanging nodes.)
    AffineConstraints<double> constraints;

    // A container for the sparse linear system which will eventually result from the weak form of our problem.
    SparseMatrix<double> system_matrix;

    // Necessary to set up the SparseMatrix.  Basically this just records where the nonzero entries are.
    SparsityPattern      sparsity_pattern;

    // A container for the solution (the vector potential in this case).
    Vector<double> solution;

    // A container for the RHS of the linear system.
    Vector<double> system_rhs;

    const double sample_sidelength;
    const double penetration_depth;
  };


  // London penetration depth
  // The vacuum is modeled by a very large value of lambda...
  // TODO: Replace this by a multi-domain implementation
  template <int dim>
  double penetration_depth(const Point<dim> &p)
  {
    const double sample_radius = 0.5;
    if (p.square() < sample_radius * sample_radius)
      return 0.1;
    else
      return 1e4;
  }

  // Class to compute Dirichlet boundary values (outer boundary).
  // Based on step-46.
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(dim)
    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
  };

  // Implementation of the "value" method for the boundary value function.
  // This will return the value of the specified component of the boundary field
  // at a given point.
  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & p,
                                          const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    if (component == 0)
      return -p(1);
    return 0;
  }

  // Returns the full boundary-value vector by calling the ::value function for
  // each component.
  template <int dim>
  void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                               Vector<double> &  values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryValues<dim>::value(p, c);
  }


  // Constructor
  // This is where we define what type of finite elements to use
  // Physical constants are also defined here
  template <int dim>
	  PoissonSolver<dim>::PoissonSolver(
		const unsigned int london_degree,
		const unsigned int maxwell_degree)
    : london_degree(london_degree)
	, maxwell_degree(maxwell_degree
	, triangulation(Triangulation<dim>::maximum_smoothing))
	, london_fe(FE_Q<dim>(london_degree),	// Type of elements to use
		dim,				// Number of components in A
		FE_Nothing<dim>(),		// 'Placeholder' elements for outside sample
		dim)				// Number of placeholders to use
    	, maxwell_fe(FE_Nothing<dim>(),		// Arguments here are similar...
		dim,
		FE_Q<dim>(maxwell_degree),
		dim) 
    	, dof_handler(triangulation)
	, penetration_depth(.01)
	, sample_sidelength(.25)
	{
	    fe_collection.push_back(london_fe);
	    fe_collection.push_back(maxwell_fe);
	}
  

  template <int dim>
  bool PoissonSolver<dim>::is_in_sample(
    const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == sample_id);
  }

  // @sect4{Meshes and assigning subdomains}

  // The next pair of functions deals with generating a mesh and making sure
  // all flags that denote subdomains are correct. <code>make_grid</code>, as
  // discussed in the introduction, generates an $8\times 8$ mesh (or an
  // $8\times 8\times 8$ mesh in 3d) to make sure that each coarse mesh cell
  // is completely within one of the subdomains. After generating this mesh,
  // we loop over its boundary and set the boundary indicator to one at the
  // top boundary, the only place where we set nonzero Dirichlet boundary
  // conditions. After this, we loop again over all cells to set the material
  // indicator &mdash; used to denote which part of the domain we are in, to
  // either the fluid or solid indicator.
  template <int dim>
  void PoissonSolver<dim>::make_grid()
  {
    GridGenerator::subdivided_hyper_cube(triangulation, 8, -1, 1);


	// Only need one boundary id for this problem 
	// Conditions are homogeneous on both boundaries
    /*    for (const auto &cell : triangulation.active_cell_iterators())*/
      //for (const auto &face : cell->face_iterators())
        //if (face->at_boundary() && (face->center()[dim - 1] == 1))
          /*face->set_all_boundary_ids(1);*/


    for (const auto &cell : dof_handler.active_cell_iterators())
      if (((std::fabs(cell->center()[0]) < 0.25) &&
           (cell->center()[dim - 1] > 0.5)) ||
          ((std::fabs(cell->center()[0]) >= 0.25) &&
           (cell->center()[dim - 1] > -0.5)))
        cell->set_material_id(sample_id);
      else
        cell->set_material_id(vacuum_id);
  }


  // The second part of this pair of functions determines which finite element
  // to use on each cell. Above we have set the material indicator for each
  // coarse mesh cell, and as mentioned in the introduction, this information
  // is inherited from mother to child cell upon mesh refinement.
  //
  // In other words, whenever we have refined (or created) the mesh, we can
  // rely on the material indicators to be a correct description of which part
  // of the domain a cell is in. We then use this to set the active FE index
  // of the cell to the corresponding element of the hp::FECollection member
  // variable of this class: zero for fluid cells, one for solid cells.
  template <int dim>
  void PoissonSolver<dim>::set_active_fe_indices()
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell_is_in_fluid_domain(cell))
          cell->set_active_fe_index(0);
        else if (cell_is_in_solid_domain(cell))
          cell->set_active_fe_index(1);
        else
          Assert(false, ExcNotImplemented());
      }
  }

  // Implementation of setup_system
  template <int dim>
  void PoissonSolver<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Dirichlet conditions: Outer boundary
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);

    // Neumann conditions: Inner boundary (at material interface)
    std::set<types::boundary_id> no_normal_flux_boundaries;
    no_normal_flux_boundaries.insert(0);
    VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                     1, /* Boundary indicator flag */
                                                     no_normal_flux_boundaries,
                                                     constraints);

    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
  }


  // This is where the "meat" of the problem is---the bilinear form representing
  // the weak form of the PDE is defined in the loops over quadrature points.
  template <int dim>
  void PoissonSolver<dim>::assemble_system()
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    // A matrix representing the system restricted to a single cell.
    // The data from this matrix will be moved into the overall system matrix on each
    // pass through the loop over cells.
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const double current_penetration_depth =
              penetration_depth<dim>(fe_values.quadrature_point(q_index));
            for (const unsigned int i : fe_values.dof_indices())
              {
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (current_penetration_depth *        // lambda
                     fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index));           // dx

                cell_rhs(i) += (1.0 *                               // f(x)
                                fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                fe_values.JxW(q_index));            // dx
              }
          }

        // Finally, transfer the contributions from @p cell_matrix and
        // @p cell_rhs into the global objects.
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }



  // Solves the sparse linear system using a conjugate-gradient method.
  // template <int dim>
  // void PoissonSolver<dim>::solve()
  // {
  //   SolverControl            solver_control(1000, 1e-12);
  //   SolverCG<Vector<double>> solver(solver_control);
  //
  //   PreconditionSSOR<SparseMatrix<double>> preconditioner;
  //   preconditioner.initialize(system_matrix, 1.2);
  //
  //   solver.solve(system_matrix, solution, system_rhs, preconditioner);
  //
  //   constraints.distribute(solution);
  // }

  // Directly solves the sparse linear system.
  // The above, commented-out implementation is preferable, but this approach will always work.
  template <int dim>
  void PoissonSolver<dim>::solve()
  {
    SparseDirectUMFPACK direct_solver;
    direct_solver.initialize(system_matrix);
    direct_solver.vmult(solution, system_rhs);
    constraints.distribute(solution);
  }


  // Applies an error estimator to determine where the mesh should be refined.
  // The error estimator used here was designed for the Laplace equation.  However,
  // it performs reasonably well for other elliptic operators.
  template <int dim>
  void PoissonSolver<dim>::refine_grid()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                       solution,
                                       estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);


    triangulation.execute_coarsening_and_refinement();
  }


  // Outputs grids and solution data for visualization.
  // The library supports many output formats, but I've stuck with VTU here
  // because it seems to be the most common choice in the deal.ii tutorials.
  template <int dim>
  void PoissonSolver<dim>::output_results(const unsigned int cycle) const
  {
    {
      GridOut               grid_out;
      std::ofstream         output("grid-" + std::to_string(cycle) + ".vtu");
      // GridOutFlags::Gnuplot gnuplot_flags(false, 5);
      // grid_out.set_flags(gnuplot_flags);
      //MappingQGeneric<dim> mapping(3);
      grid_out.write_vtu(triangulation, output /*, &mapping */);
    }

    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      data_out.build_patches();

      std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
      data_out.write_vtu(output);
    }
  }



  // This function is only responsible for making sure the other steps in the algorithm
  // execute in the correct order.
  template <int dim>
  void PoissonSolver<dim>::run()
  {
    for (unsigned int cycle = 0; cycle < 8; ++cycle)
      {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_ball(triangulation);
            triangulation.refine_global(1);
          }
        else
          refine_grid();


        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;

        setup_system();

        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        assemble_system();
        solve();
        output_results(cycle);
      }
  }



} // namespace


// Most of this function is boilerplate exception-handling and can be ignored.
int main()
{
  using namespace VectorPoissonSolver;
  using namespace dealii;

  try
    {
      PoissonSolver<3> poisson_problem_3d;
      poisson_problem_3d.run();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
