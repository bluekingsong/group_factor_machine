all: fm1

INC=-Isrc -I/home/lanjinsong/common/gflags-2.0/include
LIB=-L/home/lanjinsong/common/gflags-2.0/lib/ -lgflags
cc=/usr/bin/g++ -Wall -g -O3

fm1: src/gradient_calc.cc src/dataset.cc  src/gfm_gradient_calc.cc src/group_factor_machine.cc src/gfm_train.cc src/auc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/config.h src/lbfgs.cc  src/gradient_descent.cc src/cpp_common.cc src/online_optimizer.cc 
	${cc} -o factor_machine src/gradient_calc.cc src/dataset.cc  src/gfm_gradient_calc.cc src/group_factor_machine.cc src/gfm_train.cc src/auc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/config.h src/lbfgs.cc  src/gradient_descent.cc src/cpp_common.cc src/online_optimizer.cc ${INC} ${LIB}

fm: src/gradient_calc.cc src/dataset.cc  src/gfm_gradient_calc.cc src/group_factor_machine.cc src/gfm_train.cc src/auc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/conjugate_gradient.cc src/hessian_vec_product.cc src/config.h src/sample_lbfgs.cc src/lbfgs.cc  src/gradient_descent.cc src/cpp_common.cc
	/usr/bin/g++ -O3 src/dataset.cc src/gradient_calc.cc src/gfm_gradient_calc.cc src/group_factor_machine.cc src/gfm_train.cc src/auc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/conjugate_gradient.cc src/hessian_vec_product.cc src/config.h src/sample_lbfgs.cc src/lbfgs.cc  src/gradient_descent.cc src/cpp_common.cc -Isrc -o factor_machine

utest: src/dataset.cc  src/unittest.cc src/cpp_common.cc src/gfm_gradient_calc.cc src/vec_op.cc src/gradient_calc.cc src/auc.cc
	${cc} -o utest  src/dataset.cc  src/unittest.cc src/cpp_common.cc src/gfm_gradient_calc.cc src/vec_op.cc src/gradient_calc.cc src/auc.cc ${INC} ${LIB}

clean:
	rm -f utest factor_machine

gd: src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/gradient_descent.cc src/gd_train.cc src/vec_op.cc src/config.h src/cpp_common.cc
	/usr/bin/g++ src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/gradient_descent.cc src/gd_train.cc src/vec_op.cc src/cpp_common.cc -Isrc -o gd

sgd: src/dataset.cc src/gradient_calc.cc  src/optimizer.cc src/stochastic_gradient_descent.cc src/sgd_train.cc src/vec_op.cc src/config.h src/cpp_common.cc
	/usr/bin/g++  src/dataset.cc src/gradient_calc.cc  src/optimizer.cc src/stochastic_gradient_descent.cc src/sgd_train.cc src/vec_op.cc src/cpp_common.cc -Isrc -o sgd

lbfgs: src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/lbfgs.cc src/lbfgs_train.cc src/vec_op.cc src/gradient_descent.cc src/config.h src/cpp_common.cc
	/usr/bin/g++ src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/lbfgs.cc src/lbfgs_train.cc src/gradient_descent.cc src/cpp_common.cc -Isrc -o lbfgs

newton: src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/inexact_newton.cc src/conjugate_gradient.cc src/newton_train.cc src/hessian_vec_product.cc src/config.h src/cpp_common.cc
	/usr/bin/g++ src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/inexact_newton.cc src/conjugate_gradient.cc src/newton_train.cc src/hessian_vec_product.cc src/cpp_common.cc -Isrc -o newton

sbfgs: src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/conjugate_gradient.cc src/hessian_vec_product.cc src/config.h src/sample_lbfgs.cc src/lbfgs.cc src/sample_lbfgs_train.cc src/gradient_descent.cc src/cpp_common.cc
	/usr/bin/g++ src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/conjugate_gradient.cc src/sample_lbfgs.cc src/sample_lbfgs_train.cc src/hessian_vec_product.cc src/lbfgs.cc src/gradient_descent.cc src/cpp_common.cc -Isrc -o sbfgs

#utest: src/conjugate_gradient.cc src/gradient_descent.cc src/optimizer.cc src/vec_op.cc src/dataset.cc src/hessian_vec_product.cc src/linear_search.cc src/sample_lbfgs.cc src/inexact_newton.cc src/mat_vec_product.cc src/gradient_calc.cc src/lbfgs.cc src/unittest.cc src/cpp_common.cc
#	/usr/bin/g++ src/conjugate_gradient.cc src/gradient_descent.cc src/optimizer.cc src/vec_op.cc src/dataset.cc src/hessian_vec_product.cc src/linear_search.cc src/sample_lbfgs.cc src/inexact_newton.cc src/mat_vec_product.cc src/gradient_calc.cc src/lbfgs.cc src/unittest.cc src/cpp_common.cc -Isrc -o utest

tron: src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/inexact_newton.cc src/conjugate_gradient.cc src/tron_train.cc src/hessian_vec_product.cc src/config.h src/cpp_common.cc src/trust_region.cc
	/usr/bin/g++ src/dataset.cc src/gradient_calc.cc src/linear_search.cc src/optimizer.cc src/vec_op.cc src/inexact_newton.cc src/conjugate_gradient.cc src/tron_train.cc src/hessian_vec_product.cc src/cpp_common.cc src/trust_region.cc -Isrc -o tron
