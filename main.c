/* Cyclic Coordinate Descent inverse kinematics solver.

   Copyright (C) 2009 Dmitri Bachtin <damg.dev@googlemail.com>

   This program is free software: you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program. If not, see
   <http://www.gnu.org/licenses/>.
 */


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#define ACOS_NUM_ERROR_EPSILON 0.0000001

#ifndef NDEBUG
#define PRINTI(msg, i)				\
  printf(msg);					\
  printf("%i\n", i);				

#define PRINTD(msg, d)				\
  printf(msg);					\
  printf("%f\n", d);				

#define PRINTV(msg, v)				\
  {						\
    printf(msg);				\
    size_t i;					\
    for(i = 0; i < v->size; ++i)		\
      printf("%f ", gsl_vector_get(v, i));	\
    printf("\n");				\
  }	      

#define PRINTA(msg, a, n)			\
  {						\
    size_t i;					\
    printf(msg);				\
    for(i = 0; i < n; ++i)			\
      printf("%f ", a[i]);			\
    printf("\n");				\
  }

#define PRINTM(msg, m)\
  {						 \
  printf(msg);					 \
  size_t i,j;					 \
  for(i = 0; i < m->size1; ++i)			 \
    {						 \
      for(j = 0; j < m->size2; ++j)		 \
	printf("%f ", gsl_matrix_get(m, i, j));	 \
      printf("\n");				 \
    }						 \
  }
#else
#define PRINTI(msg, i)
#define PRINTD(msg, d)
#define PRINTV(msg, v)
#define PRINTM(msg, m)
#define PRINTA(msg, a, n)
#define DBG(...)
#endif

typedef enum 
  {
    E_AXIS_X, E_AXIS_Y, E_AXIS_Z,
  } E_AXIS;

struct ar_m_dh_params
{
  double theta, alpha, d, a;
};

struct ar_m_limb;

struct ar_m_limb
{
  struct ar_m_limb *prev, *next;
  struct ar_m_dh_params params;
  double min_theta, max_theta;
};

gsl_matrix*
ar_m_dh_trnsfmtn_mtx(struct ar_m_dh_params* dh)
{
  assert(dh != NULL);
  gsl_matrix* r;
  double sin_theta, cos_theta, sin_alpha, cos_alpha;

  sin_theta = sin(dh->theta);
  cos_theta = cos(dh->theta);
  sin_alpha = sin(dh->alpha);
  cos_alpha = cos(dh->alpha);

  r = gsl_matrix_calloc(4,4);

  gsl_matrix_set(r, 0, 0, cos_theta);
  gsl_matrix_set(r, 0, 1, -sin_theta * cos_alpha);
  gsl_matrix_set(r, 0, 2, sin_theta * sin_alpha);
  gsl_matrix_set(r, 0, 3, dh->a * cos_theta);

  gsl_matrix_set(r, 1, 0, sin_theta);
  gsl_matrix_set(r, 1, 1, cos_theta * cos_alpha);
  gsl_matrix_set(r, 1, 2, -cos_theta * sin_alpha);
  gsl_matrix_set(r, 1, 3, dh->a * sin_theta);

  gsl_matrix_set(r, 2, 1, sin_alpha);
  gsl_matrix_set(r, 2, 2, cos_alpha);
  gsl_matrix_set(r, 2, 3, dh->d);

  gsl_matrix_set(r, 3, 3, 1);

  return r;
};

struct ar_m_limb*
ar_m_limb_chain_root(struct ar_m_limb* l)
{
  assert(l != NULL);
  while(l->prev != NULL)
    l = l->prev;
  return l;
}

struct ar_m_limb*
ar_m_limb_chain_end(struct ar_m_limb* l)
{
  assert(l != NULL);
  while(l->next != NULL)
    l = l->next;
  return l;
}

double rad2deg(double rad) { return rad / M_PI * 180.0; }

size_t
ar_m_limb_chain_length(struct ar_m_limb* l)
{
  assert(l != NULL);

  size_t i = 1;
  while(l->next != NULL)
    {
      l = l->next;
      ++i;
    }

  return i;
}

gsl_matrix*
ar_m_limb_chain_trnsfmtn_mtx_n(struct ar_m_limb *l,
			       size_t n)
{
  assert(l != NULL);
  assert(ar_m_limb_chain_length(l) >= n);

  gsl_matrix *r = gsl_matrix_calloc(4,4);
  gsl_matrix_set_identity(r);
  gsl_matrix *tmp = gsl_matrix_alloc(r->size1, r->size2);
  size_t i;
  for(i = 0; i != n; ++i, l = l->next)
    {
      gsl_matrix* dh = ar_m_dh_trnsfmtn_mtx(&l->params);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, r, dh, 0.0, tmp);
      gsl_matrix_memcpy(r, tmp);
      gsl_matrix_free(dh);
    }
  gsl_matrix_free(tmp);
  return r;
}

gsl_matrix*
ar_m_limb_chain_trnsfmtn_mtx_all(struct ar_m_limb* l)
{
  return ar_m_limb_chain_trnsfmtn_mtx_n(l, ar_m_limb_chain_length(l));
}

gsl_vector*
ar_m_dh_pos(const gsl_matrix* dh)
{
  gsl_vector* r = gsl_vector_alloc(3);
  gsl_vector_set(r, 0, gsl_matrix_get(dh, 0, 3));
  gsl_vector_set(r, 1, gsl_matrix_get(dh, 1, 3));
  gsl_vector_set(r, 2, gsl_matrix_get(dh, 2, 3));
  
  return r;
}

gsl_vector*
gsl_vector_cross(const gsl_vector* v1,
		 const gsl_vector* v2)
{
  assert(v1 != NULL);
  assert(v2 != NULL);
  assert(v1->size == 3);
  assert(v2->size == 3);

  gsl_vector *r = gsl_vector_alloc(3);
  double x1,y1,z1,x2,y2,z2;

  x1 = gsl_vector_get(v1, 0);
  y1 = gsl_vector_get(v1, 1);
  z1 = gsl_vector_get(v1, 2);
  
  x2 = gsl_vector_get(v2, 0);
  y2 = gsl_vector_get(v2, 1);
  z2 = gsl_vector_get(v2, 2);

  gsl_vector_set(r, 0, y1*z2 - z1*y2);
  gsl_vector_set(r, 1, z1*x2 - x1*z2);
  gsl_vector_set(r, 2, x1*y2 - y1*x2);

  return r;
}

double
gsl_vector_length_sq(const gsl_vector *v)
{
  assert(v != NULL);

  double sum = 0.0;
  size_t i;
  double x;
  for(i = 0; i < v->size; ++i)
    {
      x = gsl_vector_get(v, i);
      sum += x*x;
    }
  return sum;
}

double
gsl_vector_length(const gsl_vector* v)
{
  return sqrt(gsl_vector_length_sq(v));
}

double
gsl_vector_distance_sq(const gsl_vector* v1,
		       const gsl_vector* v2)
{
  assert(v1 != NULL);
  assert(v2 != NULL);
  assert(v1->size == v2->size);
  
  gsl_vector *diff = gsl_vector_alloc(v1->size);
  gsl_vector_memcpy(diff, v1);
  gsl_vector_sub(diff, v2);
  double len = gsl_vector_length_sq(diff);
  gsl_vector_free(diff);
  return len;
}

double
gsl_vector_distance(const gsl_vector* v1,
		    const gsl_vector* v2)
{
  return sqrt(gsl_vector_distance_sq(v1, v2));
}

double gsl_vector_dot(const gsl_vector *v1,
		      const gsl_vector *v2)
{
  assert(v1 != NULL);
  assert(v2 != NULL);
  assert(v1->size == v2->size);

  double sum = 0.0, a, b;
  size_t i;
  for(i = 0; i < v1->size; ++i)
    {
      a = gsl_vector_get(v1, i);
      b = gsl_vector_get(v2, i);
      sum += (a*b);
    }
  return sum;
}

void gsl_vector_normalize(gsl_vector* v)
{
  double len = gsl_vector_length(v);
  gsl_vector_scale(v, 1.0 / len);
}

// performs one descent operation
bool ccd_chain_iter(const gsl_vector* goal_pos,
		    struct ar_m_limb* chain,
		    double epsilon)
{
  assert(goal_pos != NULL);
  assert(chain != NULL);
  assert(epsilon > 0);

  size_t chain_len = ar_m_limb_chain_length(chain);
  size_t root_index;
  gsl_matrix *end_effector_dh, *root_dh;
  gsl_vector *end_effector_pos, *root_pos;
  gsl_vector *root_to_end_effector = gsl_vector_alloc(3);
  gsl_vector *root_to_goal = gsl_vector_alloc(3);
  gsl_vector *plane_root_to_end_effector = gsl_vector_alloc(3);
  gsl_vector *plane_root_to_goal = gsl_vector_alloc(3);

  bool stop = false;

  // from each chain node we have to rotate the root->end_effector
  // vector towards the root->goal vector.
  for(root_index = chain_len - 1; root_index != -1; --root_index)
    {
      // calculate end effector and root positions
      PRINTI("Root index: ", root_index);
      end_effector_dh = ar_m_limb_chain_trnsfmtn_mtx_all(chain);
      root_dh = ar_m_limb_chain_trnsfmtn_mtx_n(chain, root_index);
      end_effector_pos = ar_m_dh_pos(end_effector_dh);
      root_pos = ar_m_dh_pos(root_dh);

      PRINTM("root dh:\n", root_dh);
      PRINTM("end effector dh:\n", end_effector_dh);
      PRINTV("root position: ", root_pos);
      PRINTV("end effector position: ", end_effector_pos);

      // if both positions are near enough, we can stop
      double distance = gsl_vector_distance(end_effector_pos, goal_pos);
      PRINTD("distance: ", distance);
      if (distance <= epsilon)
	{
	  gsl_vector_free(root_to_end_effector);
	  gsl_vector_free(root_to_goal);
	  gsl_vector_free(plane_root_to_end_effector);
	  gsl_vector_free(plane_root_to_goal);
	  stop = true;
	}
      else // positions are not near enough, optimize
	{
	  // calculate root->end_effector and root->goal vectors
	  // the former has to be rotated towards the latter
	  gsl_vector_memcpy(root_to_end_effector, end_effector_pos);
	  gsl_vector_sub(root_to_end_effector, root_pos);
	  
	  gsl_vector_memcpy(root_to_goal, goal_pos);
	  gsl_vector_sub(root_to_goal, root_pos);
	  
	  // if the root is directly on goal, we cannot do anything
	  // proceed with the next one and prey
	  if (gsl_vector_length_sq(root_to_goal) != 0.0)
	    {	  
	      PRINTV("root to end effector: ", root_to_end_effector);
	      PRINTV("root to goal: ", root_to_goal);
	      
	      // coordinate transformation matrix.
	      // both vectors have to be brought into a uniform coordinate system
	      // where they can be rotated around an axis (here: Z)
	      gsl_matrix_view root_rot_m = gsl_matrix_submatrix(root_dh, 0, 0, 3, 3);
	      gsl_blas_dgemv(CblasNoTrans, 1.0, &root_rot_m.matrix, root_to_end_effector, 0.0, plane_root_to_end_effector);
	      gsl_blas_dgemv(CblasNoTrans, 1.0, &root_rot_m.matrix, root_to_goal, 0.0, plane_root_to_goal);
	      
	      PRINTV("plane root to end effector: ", plane_root_to_end_effector);
	      PRINTV("plane root to goal: ", plane_root_to_goal);
	      
	      // before computing the angle the vectors have to be
	      // brought on the same plane and normalized.
	      // as they are along z-axis now, simply make both z
	      // components same.
	      gsl_vector_set(plane_root_to_goal, 2, gsl_vector_get(plane_root_to_end_effector, 2));
	      gsl_vector_normalize(plane_root_to_end_effector);
	      gsl_vector_normalize(plane_root_to_goal);
	      
	      PRINTV("normalized plane root to end effector: ", plane_root_to_end_effector);
	      PRINTV("normalized root to goal: ", plane_root_to_goal);
	      
	      // cos(theta) = dot_product(v1, v2)
	      // due to numerical errors this value is sometimes
	      // bigger than 1.0 eliminate this.
	      double cos_dtheta = gsl_vector_dot(plane_root_to_goal, plane_root_to_end_effector);
	      if (cos_dtheta > 1.0 && cos_dtheta - ACOS_NUM_ERROR_EPSILON <= 1.0) // eliminate numerical error 
		cos_dtheta = 1.0;
	      PRINTD("cos dtheta: ", cos_dtheta);
	      // the rotation angle
	      double dtheta = acos(cos_dtheta);
	      PRINTD("delta theta: ", dtheta);
	      PRINTD("delta theta deg: ", rad2deg(dtheta));
	      
	      // cross product is needed to determine the rotation
	      // direction.
	      gsl_vector* x = gsl_vector_cross(plane_root_to_goal, plane_root_to_end_effector);
	      double sign = gsl_vector_get(x, 2) < 0 ? 1.0 : -1.0;
	      gsl_vector_free(x);
	      
	      // find the root node and rotate its angle by dtheta.
	      struct ar_m_limb* root = chain;
	      size_t i;
	      for(i = 0; i != root_index; ++i)
		root = root->next;
	      PRINTD("current root theta: ", root->params.theta);
	      PRINTD("current root theta deg: ", rad2deg(root->params.theta));
	      root->params.theta += sign*dtheta;
	      if (root->params.theta > root->max_theta)
		root->params.theta = root->max_theta;
	      if (root->params.theta < root->min_theta)
		root->params.theta = root->min_theta;
	      PRINTD("new root theta: ", root->params.theta);
	      PRINTD("new root theta deg: ", rad2deg(root->params.theta));
	    }
	  gsl_vector_free(end_effector_pos);
	  gsl_vector_free(root_pos);
	  gsl_matrix_free(end_effector_dh);
	  gsl_matrix_free(root_dh);
	}
      if (stop)
	return true;
    }
  
  return false;
}

// kicks in the ccd_iter algorithm max_iterations times.
bool ccd(const gsl_vector* goal_pos,
	 struct ar_m_limb *chain,
	 double epsilon,
	 size_t max_iterations)
{
  size_t iteration;
  for(iteration = 0; iteration < max_iterations; ++iteration)
    {
      PRINTI("iteration: ", iteration);
      if (ccd_chain_iter(goal_pos, chain, epsilon))
	return true;
    }
  return false;
}

int main()
{
  struct ar_m_dh_params p1 = { M_PI/2, 0, 0, 1 };
  struct ar_m_dh_params p2 = { 0.0, 0, 0, 1 };
  struct ar_m_dh_params p3 = { 0.0, 0, 0, 1 };
  struct ar_m_dh_params p4 = { 0.0, 0, 0, 1 };

  struct ar_m_limb l1 = { NULL, NULL, p1, -M_PI/2.0, M_PI/2 };
  struct ar_m_limb l2 = { NULL, NULL, p2, -M_PI/2.0, M_PI/2 };
  struct ar_m_limb l3 = { NULL, NULL, p3, -M_PI/2.0, M_PI/2 };
  struct ar_m_limb l4 = { NULL, NULL, p4, -M_PI/2.0, M_PI/2 };

  l1.next = &l2;
  l2.next = &l3;
  l3.next = &l4;
  
  l2.prev = &l1;
  l3.prev = &l2;
  l4.prev = &l3;

  struct ar_m_limb *limbs[] = { &l1,  &l2, &l3, &l4 };
  size_t i;

  double agoal[] = { 1.0, 2.0, 0.0 };
  PRINTA("goal: ", agoal, 3);

  gsl_vector_const_view goal = gsl_vector_const_view_array(agoal, 3);
  bool rc = ccd(&goal.vector, &l1, 0.001, 100);

  gsl_matrix *dh = ar_m_limb_chain_trnsfmtn_mtx_all(&l1);
  gsl_vector* pos = ar_m_dh_pos(dh);
  
  PRINTV("end position: ", pos);
  PRINTI("success: ", rc);
  gsl_matrix_free(dh);
  gsl_vector_free(pos);

  return 0;
}
