from assignment.simulator import StatevectorSimulator

import numpy as np
from numpy.typing import NDArray

MAX_QUBITS = 5
RANDOM_SEED = 42
RANDOM_ITERATIONS = 10
SHOTS = 8192

class TestStatevectorSimulator:


    def reference_zero_state_reference(self, num_qubits: int) -> NDArray[np.complex128]:
        solution = np.zeros(2**num_qubits, dtype=np.complex128)
        solution[0] = 1
        return solution
    
    
    def test_zero_state(self) -> None:
        simulator = StatevectorSimulator()
        for num_qubits in range(1, MAX_QUBITS + 1):
            answer = simulator.create_zero_state(num_qubits)
    
            assert answer.shape == (2**num_qubits,)
    
            solution = self.reference_zero_state_reference(num_qubits)
    
            assert np.allclose(solution, answer)
    
    
    def reference_h_gate_on_target(self, num_qubits: int, target: int) -> NDArray[np.complex128]:
        h_matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        solution = h_matrix if target == 0 else np.eye(2)
        for site in range(1, num_qubits):
            solution = np.kron(h_matrix, solution) if site == target else np.kron(np.eye(2), solution)
        return solution
    
    
    def test_h_gate(self) -> None:
        h_matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        simulator = StatevectorSimulator()
        for num_qubits in range(1, MAX_QUBITS + 1):
            for target in range(num_qubits):
                answer = simulator.create_single_qubit_gate(num_qubits, target, h_matrix)
    
                assert answer.shape == (2**num_qubits, 2**num_qubits)
    
                solution = self.reference_h_gate_on_target(num_qubits, target)
    
                assert np.allclose(solution, answer)
    
    
    def reference_nn_gate_on_targets(self, num_qubits: int, qubit0: int, qubit1: int) -> NDArray[np.complex128]:
        gate_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.complex128)
        identity = np.eye(2)
        if qubit0 == 0:
            gate = gate_matrix
            for _ in range(2, num_qubits):
                gate = np.kron(identity, gate)
            return gate
    
        gate = identity
        for _ in range(1, qubit0):
            gate = np.kron(identity, gate)
    
        gate = np.kron(gate_matrix, gate)
    
        for _ in range(qubit1 + 1, num_qubits):
            gate = np.kron(identity, gate)
    
        return gate
    
    
    def test_nn_gate(self) -> None:
        gate_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.complex128)
        simulator = StatevectorSimulator()
        for num_qubits in range(2, MAX_QUBITS + 1):
            for qubit0 in range(num_qubits - 1):
                qubit1 = qubit0 + 1
                answer = simulator.create_nearest_neighbor_two_qubit_gate(num_qubits, qubit0, qubit1, gate_matrix)
    
                assert answer.shape == (2**num_qubits, 2**num_qubits)
    
                solution = self.reference_nn_gate_on_targets(num_qubits, qubit0, qubit1)
    
                assert np.allclose(solution, answer)
    
    
    def reference_controlled_gate_on_targets(self, num_qubits: int, control: int, target: int) -> NDArray[np.complex128]:
        gate_matrix = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        result = np.identity(2**num_qubits, dtype=np.complex128)
        control_bit = 1 << control
        target_bit = 1 << target
        for i in range(2**num_qubits):
            for j in range(2**num_qubits):
                x_idx = i & control_bit
                y_idx = j & control_bit
                if not x_idx or not y_idx:
                    continue
    
                if (i & ~control_bit & ~target_bit) != (j & ~control_bit & ~target_bit):
                    continue
    
                x_idx = (i & target_bit) >> target
                y_idx = (j & target_bit) >> target
    
                result[i, j] = gate_matrix[x_idx, y_idx]
    
        return result
    
    
    def test_controlled_gate(self) -> None:
        gate_matrix = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        simulator = StatevectorSimulator()
        for num_qubits in range(2, MAX_QUBITS + 1):
            for control in range(num_qubits):
                for target in range(num_qubits):
                    if control == target:
                        continue
    
                    answer = simulator.create_controlled_single_qubit_gate(num_qubits, control, target, gate_matrix)
    
                    assert answer.shape == (2**num_qubits, 2**num_qubits)
    
                    solution = self.reference_controlled_gate_on_targets(num_qubits, control, target)
    
                    assert np.allclose(solution, answer)
    
    
    def reference_apply_gate(self, state: NDArray[np.complex128], gate: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return gate @ state
    
    
    def test_apply_gate(self) -> None:
        gen = np.random.Generator(np.random.PCG64(RANDOM_SEED))
        simulator = StatevectorSimulator()
        for num_qubits in range(1, MAX_QUBITS + 1):
            for _ in range(RANDOM_ITERATIONS):
                state = gen.random((2**num_qubits,))
                gate = gen.random((2**num_qubits, 2**num_qubits))
                answer = simulator.apply_gate(state, gate)
    
                assert answer.shape == (2**num_qubits,)
    
                solution = self.reference_apply_gate(state, gate)
    
                assert np.allclose(solution, answer)
    
    
    def reference_get_amplitude(self, state: np.ndarray, bitstring: str) -> np.complex128:
        idx = int(bitstring, 2)
        return state[idx]
    
    
    def test_get_amplitude(self) -> None:
        gen = np.random.Generator(np.random.PCG64(RANDOM_SEED))
        simulator = StatevectorSimulator()
        for num_qubits in range(1, MAX_QUBITS + 1):
            for _ in range(RANDOM_ITERATIONS):
                state = gen.random((2**num_qubits,))
                state /= np.linalg.norm(state)
    
                for idx in range(2**num_qubits):
                    bitstring = format(idx, f"0{num_qubits}b")
                    answer = simulator.get_amplitude(state, bitstring)
    
                    assert answer.shape == ()
    
                    solution = self.reference_get_amplitude(state, bitstring)
    
                    assert np.allclose(solution, answer)
    
    
    def test_measure_all(self) -> None:
        gen = np.random.Generator(np.random.PCG64(RANDOM_SEED))
        simulator = StatevectorSimulator()
        for num_qubits in range(1, MAX_QUBITS + 1):
            for _ in range(RANDOM_ITERATIONS):
                state = gen.random((2**num_qubits,))
                state /= np.linalg.norm(state)
    
                answer = simulator.measure_all(state, SHOTS)
    
                for key in answer:
                    assert np.allclose(answer[key] / SHOTS, state[int(key, 2)] ** 2, atol=0.1)
    
    
    def test_simulate_ghz_state(self) -> None:
        simulator = StatevectorSimulator()
        for num_qubits in range(2, MAX_QUBITS + 1):
            answer = simulator.simulate_ghz_state(num_qubits, SHOTS)
    
            assert len(answer) == 2
    
            assert "0" * num_qubits in answer
            assert "1" * num_qubits in answer
    
            assert np.allclose(answer["0" * num_qubits] / SHOTS, 0.5, atol=0.1)
            assert np.allclose(answer["1" * num_qubits] / SHOTS, 0.5, atol=0.1)
