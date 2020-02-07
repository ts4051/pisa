"""
Utilities for performing some not-so-common matrix tasks.
"""

import numpy as np
import scipy.linalg as lin

__all__ = [
    'is_psd',
    'fronebius_nearest_psd',
]

__author__ = 'A. Trettin'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''

def is_psd(m):
    '''Test whether a matrix is positive semi-definite.

    Test is done via attempted Cholesky decomposition as suggested in:

    N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    '''
    try:
        _ = np.linalg.cholesky(m)
        return True
    except np.linalg.LinAlgError:
        return False

def fronebius_nearest_psd(A, return_distance=False):
    '''
    Find the positive semi-definite matrix that is nearest to A as measured by
    the Fronebius norm.

    This function is a modification of [1], which is a Python adaption of [2], which
    credits [3].

    [1] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    Args:
        m : numpy.ndarray
            Symmetric matrix
        return_distance : bool, optional
            Return distance of the input matrix to the approximation as given in
            theorem 2.1 in https://doi.org/10.1016/0024-3795(88)90223-6
            This can be compared to the actual Frobenius norm between the
            input and output to verify the calculation.
    '''
    assert A.ndim == 2, "input is not a 2D matrix"
    B = (A + A.T)/2.
    _, H = lin.polar(B)
    X = (B + H)/2.
    # small numerical errors can make matrices that are not exactly
    # symmetric, fix that
    X = (X + X.T)/2.
    # due to numerics, it's possible that the matrix is _still_ not psd.
    # We can fix that iteratively by adding small increments of the identity matrix.
    # This part comes from [1].
    if not is_psd(X):
        spacing = np.spacing(lin.norm(X))
        I = np.eye(X.shape[0])
        k = 1
        while not is_psd(X):
            mineig = np.min(np.real(lin.eigvals(X)))
            X += I * (-mineig * k**2 + spacing)
            k += 1
    if return_distance:
        C = (A - A.T)/2.
        lam = lin.eigvalsh(B)
        dist = np.sqrt(np.sum(lam**2, where=lam < 0.) + lin.norm(C, ord='fro')**2)
        return X, dist
    return X

def test_frob_psd(m):
    '''Test approximation of Frobenius-closest PSD on given matrix'''
    x, xdist = fronebius_nearest_psd(m, return_distance=True)
    is_psd_after = is_psd(x)
    actual_dist = lin.norm(m - x, ord='fro')
    assert is_psd_after, "did not produce PSD matrix"
    assert np.isclose(xdist, actual_dist), "actual distance differs from expectation"

if __name__ == '__main__':
    m_test = np.array([[1, -1], [2, 4]])
    test_frob_psd(m_test)
    print('matrix before:')
    print(m_test)
    print('matrix after:')
    print(fronebius_nearest_psd(m_test))
    print('The result matrix is psd and the Frobenius norm matches the expectation!')
    print('testing random matrices...')
    for i in range(100):
        m_test = np.random.randn(3, 3)
        test_frob_psd(m_test)
    print('Test passed!')
