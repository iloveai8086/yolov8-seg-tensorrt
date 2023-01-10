
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <functional>
#include "matrix.hpp"
#include <openblas/cblas.h>

Matrix::Matrix(){}
Matrix::Matrix(int rows, int cols, const std::initializer_list<float>& pdata){
    this->rows_ = rows;
    this->cols_ = cols;
    this->data_ = pdata;

    if(this->data_.size() < rows * cols)
        this->data_.resize(rows * cols);
}

Matrix::Matrix(int rows, int cols, const std::vector<float>&v){
    this->rows_ = rows;
    this->cols_ = cols;
    this->data_ = v;

    if(this->data_.size() < rows * cols)
        this->data_.resize(rows * cols);
}

Matrix Matrix::element_wise(const std::function<float(float)>& func) const{
    Matrix output(*this);
    for (int i = 0; i < output.data_.size();i++){
        output.data_[i] = func(output.data_[i]);
    }
    return output;
}

Matrix Matrix::operator*(float value) const{
    return element_wise([&](float x) -> float{return x * value;});
}
Matrix Matrix::operator+(float value) const{
    return element_wise([&](float x) -> float{return x + value;});
}

Matrix Matrix::operator/(float value) const{
    return element_wise([&](float x) -> float{return x / value;});
}

Matrix Matrix::operator-(float value) const{
    return element_wise([&](float x) -> float{return x - value;});
}

Matrix Matrix::operator*(const Matrix& value) const{
    Matrix output(*this);
    auto pleft = output.data_.data();
    auto pright = value.data_.data();
    for (int i = 0; i < output.data_.size(); i++)
        *pleft++ *= *pright++;
    return output;
}

Matrix Matrix::view(int rows, int cols) const{
    if(rows * cols != this->rows_ * this->cols_){
        printf("Invalid view to %d x %d\n", rows, cols);
        return Matrix();
    }
    Matrix newmat = *this;
    newmat.rows_ = rows;
    newmat.cols_ = cols;
    return newmat;
}

Matrix Matrix::exp(float value){
    Matrix m = *this;
    for (int i = 0; i < m.data_.size();i++){
        m.data_[i] = std::exp(value * m.data_[i]);
    }
    return m;
}


Matrix Matrix::power(float y) const{
    Matrix output = *this;
    auto p0 = output.ptr();
    for(int i = 0; i < output.data_.size(); ++i, ++p0)
        *p0 = std::pow(*p0, y);
    return output;
}

float Matrix::reduce_sum() const{
    auto p0 = this->ptr();
    float output = 0;
    for(int i = 0; i < this->data_.size(); ++i)
        output += *p0++;
    return output;
}


std::ostream& operator << (std::ostream& out, const Matrix& m){

    for(int i = 0; i < m.rows(); ++i){
        for(int j = 0; j < m.cols(); ++j){
            out << m(i, j) << "\t";
        }
        out << "\n";
    }
    return out;
}


Matrix gemm(const Matrix& a, bool ta, const Matrix& b, bool tb, float alpha, float beta){

    int a_elastic_rows = ta ? a.cols() : a.rows();   /* 如果转置，则维度转过来 */
    int a_elastic_cols = ta ? a.rows() : a.cols();   /* 如果转置，则维度转过来 */
    int b_elastic_rows = tb ? b.cols() : b.rows();   /* 如果转置，则维度转过来 */
    int b_elastic_cols = tb ? b.rows() : b.cols();   /* 如果转置，则维度转过来 */

    /* c是转置后维度的行和列 */
    Matrix c(a_elastic_rows, b_elastic_cols);

    int m = a_elastic_rows;
    int n = b_elastic_cols;
    int k = a_elastic_cols;
    int lda = a.cols();
    int ldb = b.cols();
    int ldc = c.cols();

    /* cblas的gemm调用风格，在以后也会存在 */
    cblas_sgemm(
            CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
            m, n, k, alpha, a.ptr(), lda, b.ptr(), ldb, beta, c.ptr(), ldc
    );
    return c;
}

Matrix mygemm(const Matrix& a, const Matrix& b){

    Matrix c(a.rows(), b.cols());
    for(int i = 0; i < c.rows(); ++i){
        for(int j = 0; j < c.cols(); ++j){
            float summary = 0;
            for(int k = 0; k < a.cols(); ++k)
                summary += a(i, k) * b(k, j);

            c(i, j) = summary;
        }
    }
    return c;
}
