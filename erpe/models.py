from pygsti.models.modelconstruction import create_explicit_model_from_expressions, create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti import unitary_to_pauligate
from scipy.linalg import expm
from pygsti import sigmax, sigmay, sigmaz, sigmazz, sigmazi, sigmaiz
import numpy as np

pauli_basis_2q = [
    np.kron(np.eye(2), sigmax),
    np.kron(np.eye(2), sigmay),
    np.kron(np.eye(2), sigmaz),
    np.kron(sigmax, np.eye(2)),
    np.kron(sigmax, sigmax),
    np.kron(sigmax, sigmay),
    np.kron(sigmax, sigmaz),
    np.kron(sigmay, np.eye(2)),
    np.kron(sigmay, sigmax),
    np.kron(sigmay, sigmay),
    np.kron(sigmay, sigmaz),
    np.kron(sigmaz, np.eye(2)),
    np.kron(sigmaz, sigmax),
    np.kron(sigmaz, sigmay),
    np.kron(sigmaz, sigmaz)
]




def X90_unitary(x_error, z_error):
    """Definition of an Xpi2 rotation with x and z errors

    The gate is 
    X = exp(-i/2 *( (pi/2 + x_error) * X + z_error * Z ) )

    Args:
        x_error (angle): x error angle.
        z_error (angle): z error angle.

    Returns:
        pygsti.unitary_to_pauligate: X gate.
    """
    H = (np.pi/2 + x_error) * sigmax + z_error * sigmaz
    return expm(-(1j/2) * H)

def X90_unitary_model2(overrot_error, axis_error):
    """Definition of an Xpi2 rotation with x and z errors

    The gate is 
    X = exp(-i/2 *( (pi/2 + x_error) * X + z_error * Z ) )

    Args:
        x_error (angle): x error angle.
        z_error (angle): z error angle.

    Returns:
        pygsti.unitary_to_pauligate: X gate.
    """
    H = (np.pi/2)*(1 + overrot_error)*(np.cos(axis_error) * sigmax + np.sin(axis_error) * sigmaz)
    return expm(-(1j/2) * H)

def Z90_unitary(phase_error):
    """Definition of a Zpi2 rotation with a phase error

    The gate is
    Z = exp(-i/2 * (pi/2 + phase_error) * Z)

    Args:
        phase_error (angle): phase error angle.

    Returns:
        pygsti.unitary_to_pauligate: Z gate.
    """
    H = (np.pi/2 + phase_error) * sigmaz
    return expm(-(1j/2) * H)

def Idle_unitary(phase_error):
    """Definition of an idle gate with a phase error

    The gate is
    I = exp(-i/2 * phase_error * Z)

    Args:
        phase_error (angle): phase error angle.

    Returns:
        pygsti.unitary_to_pauligate: I gate.
    """
    H = phase_error * sigmaz
    return expm(-(1j/2) * H)

def CZ_unitary(x_IZ, x_ZI, x_ZZ):
    """Definition of a CZ gate with three phase errors
    """
    H0 = np.pi/2 * (sigmazi + sigmaiz) - np.pi/2 * (sigmazz + np.eye(4))
    Hdelta = x_ZI * sigmazi + x_IZ * sigmaiz + x_ZZ * sigmazz
    return expm(-(1j/2) * (H0 + Hdelta))

def idle_unitary_2q_general(xvec):
    assert len(xvec) == 15
    H = sum([xvec[i] * pauli_basis_2q[i] for i in range(15)])
    return expm(-(1j/2) * H)



def create_XZ_model_1q(x_vec, 
                    qid: str='Q0', 
                    gate_depol_rate : float=0., 
                    spam_depol_rate : float=0.):
    """
    Create a single-qubit model with X and Z rotations.

    Parameters
    ----------
    x_vec : numpy.ndarray
        3-element array specifying the error parameters 
        (x_error, z_error, phase_error)

    qid : str, optional
        The qubit id. Default is 'Q0'.

    gate_depol_rate : float, optional
        The depolarizing error rate for the gates. Default is 0.
    Returns
    -------
    Model
    """
    x_error, z_error, phase_error = x_vec
    # Define the model
    model = create_explicit_model_from_expressions(
        [qid], ['Gi'],
        [f"I({qid})"],
        evotype="densitymx", )
    
    # Add the X90 and Y90 gates
    model.operations[('Gxpi2', qid)] = unitary_to_pauligate(X90_unitary(x_error, z_error))
    model.operations[('Gzpi2', qid)] = unitary_to_pauligate(Z90_unitary(phase_error))

    if gate_depol_rate > 0: 
        model.operations[('Gxpi2', qid)].depolarize(gate_depol_rate)
        model.operations[('Gzpi2', qid)].depolarize(gate_depol_rate)
    model.sim = 'map'
    return model

def create_XI_model_1q(x_vec, 
                    qid: str='Q0', 
                    gate_depol_rate : float=0., 
                    spam_depol_rate : float=0.):
    """
    Create a single-qubit model with X and I errors.

    Parameters
    ----------
    x_vec : numpy.ndarray
        3-element array specifying the error parameters 
        (x_error, detuning, idle_phase_error

    qid : str, optional
        The qubit id. Default is 'Q0'.

    gate_depol_rate : float, optional
        The depolarizing error rate for the gates. Default is 0.
    Returns
    -------
    Model
    """
    x_error, detuning, phase_error = x_vec
    # Define the model
    model = create_explicit_model_from_expressions(
        [qid], ['Gi'],
        [f"I({qid})"],
        evotype="densitymx", )
    
    # Add the X90 and Y90 gates
    model.operations[('Gxpi2', qid)] = unitary_to_pauligate(X90_unitary_model2(x_error, detuning))
    model.operations[('Gzpi2', qid)] = unitary_to_pauligate(Z90_unitary(0))
    model.operations[('Gi', qid)] = unitary_to_pauligate(Idle_unitary(phase_error))

    if gate_depol_rate > 0: 
        model.operations[('Gxpi2', qid)].depolarize(gate_depol_rate)
        model.operations[('Gi', qid)].depolarize(gate_depol_rate)
    model.sim = 'map'
    return model


def create_CZ_model(x_vec, 
                    qids: list=['Q0', 'Q1'],
                    single_qb_depol_rate : float=0., 
                    two_qb_depol_rate : float=0.):
    """
    Create a single-qubit model with X and I errors.

    Parameters
    ----------
    x_vec : numpy.ndarray
        3-element array specifying the error parameters 
        (x_error, detuning, idle_phase_error

    qid : str, optional
        The qubit id. Default is 'Q0'.

    gate_depol_rate : float, optional
        The depolarizing error rate for the gates. Default is 0.
    Returns
    -------
    Model
    """
    U = CZ_unitary(*x_vec)

    pspec = QubitProcessorSpec(num_qubits=2, gate_names=['Gxpi2', 'Gypi2', 'Gzpi2', 'Gcz'],
                                nonstd_gate_unitaries={'Gcz': U},
                                qubit_labels = qids,
                                availability={'Gcz': [tuple(qids), ]})
    model = create_explicit_model(pspec)
    model.sim = 'map'
    return model

def create_idle_model(x_vec, 
                    qids: list=['Q0', 'Q1'],
                    single_qb_depol_rate : float=0., 
                    two_qb_depol_rate : float=0.):
    """
    Create a single-qubit model with X and I errors.

    Parameters
    ----------
    x_vec : numpy.ndarray
        3-element array specifying the error parameters 
        (x_error, detuning, idle_phase_error

    qid : str, optional
        The qubit id. Default is 'Q0'.

    gate_depol_rate : float, optional
        The depolarizing error rate for the gates. Default is 0.
    Returns
    -------
    Model
    """
    Uidle = idle_unitary_2q_general(x_vec)

    pspec = QubitProcessorSpec(num_qubits=2, gate_names=['Gxpi2', 'Gypi2', 'Gzpi2', 'Gcphase', 'Gidle'],
                                nonstd_gate_unitaries={'Gidle': Uidle},
                                qubit_labels = qids,
                                availability={
                                    'Gcphase': [tuple(qids), ],
                                    'Gidle': [tuple(qids), ]
                                })
    model = create_explicit_model(pspec)
    model.sim = 'map'
    return model