from minorminer import find_embedding
import numpy as np
from dwave import embedding
from collections import defaultdict
import time
import dimod

def find_chain_strength(Q, prefactor, method):
    """
    Function to find chain strength using different methods.
    
    Args:
        Q (dict): The QUBO matrix representing the problem.
        prefactor (float): The prefactor value for chain strength optimization.
        method (str): The method to use for chain strength optimization. 
            Currently supports "UTC" (Uniform Torque Compensation) and "scaled".
    
    Returns:
        float: The chain strength value.
    
    Raises:
        ValueError: If an unsupported chain strength optimization method is provided.
    """
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    if method == "UTC":
        from dwave.embedding.chain_strength import uniform_torque_compensation
        return uniform_torque_compensation(bqm, prefactor=prefactor)
    elif method == "scaled":
        from dwave.embedding.chain_strength import scaled
        return scaled(bqm, prefactor=prefactor)
    else:
        raise ValueError(f"Unsupported chain strength method: {method}")

def get_qubit_list(embedding):
    """
    Utility function to get the list of qubits involved in an embedding.

    Parameters:
    embedding (dict): A dictionary representing the embedding of logical qubits onto physical qubits.

    Returns:
    list: A list of qubits involved in the embedding.
    """
    out = []
    for a in embedding:
        out += embedding[a]
    return out

def embedding_search(hardware_connectivity, QuboList, chain_strategy):
    """
    Performs embedding search with space between embeddings for a given hardware connectivity, QuboList, and level.

    Parameters:
    hardware_connectivity (networkx.Graph): The connectivity of the hardware.
    QuboList (list): A list of Qubo dictionaries.

    Returns:
    tuple: A tuple containing the following:
        - embeddings (dict): A dictionary of embeddings.
        - TotalQubo (dict): The total Qubo after embedding.
        - Qubo_list (list): A list of embedded Qubos.
        - offset_list (list): A list of offset values.
        - Qubits (list): A list of used qubits.
        - Qubo_without_embedding (dict): The Qubo without embedding.
        - emb_wo_chain (list): A list of embeddings without chains.
    """
    embeddings = {}
    embedding_list = []
    Qubits = []
    TotalQubo = defaultdict(float)
    offset_list = [0]
    Qubo_list = []
    Qubo_logic = {}
    
    for i in range(len(QuboList)):
        max_index = offset_list[-1]
        
        # Update the Qubo with max_index
        Qubo_temp = {}
        for (p1, p2), element in QuboList[i].items():
            Qubo_temp[(p1 + max_index, p2 + max_index)] = element

        # Perform the embedding search
        emb = find_embedding(Qubo_temp, hardware_connectivity, tries=100, max_no_improvement=100, chainlength_patience=100, timeout=100, threads=100)
        embedding_list.append(emb)

        if emb:
            # Find the chain strength for the current Qubo
            if chain_strategy == "utc":
                chain_strength = find_chain_strength(Qubo_temp, prefactor = 0.5, method = "UTC")
            else:
                chain_strength = find_chain_strength(Qubo_temp, prefactor = 1.5, method = "scaled")

            # Save used qubits
            Qubits.append(get_qubit_list(emb))

            # Create the embedded Qubo and update the total Qubo
            physical_subgraph = hardware_connectivity.subgraph(get_qubit_list(emb))
            embedded_qubo = embedding.embed_qubo(Qubo_temp, emb, physical_subgraph, chain_strength = chain_strength)
            scaled_embedded_qubo = auto_scale_qubo(embedded_qubo)
            
            max_key = max(QuboList[i].keys(), key=lambda x: max(x))
            max_index += max(max_key) + 1
            offset_list.append(max_index)
            Qubo_logic.update(Qubo_temp)
            TotalQubo = {**TotalQubo, **scaled_embedded_qubo}
            Qubo_list.append(scaled_embedded_qubo)
            emb_copy = emb.copy()
            emb = {}

            # Collect nodes in each chain and their neighbors to create a buffer
            nodes_to_remove = []
            for indx, chain in emb_copy.items():
                emb[int(indx)] = chain
                # used_nodes.extend(chain)
                for node in chain:
                    nodes_to_remove.append(node)
                    # if hardware_connectivity.has_node(node):  # Check if the node exists in the graph
                    #     for neighbor in list(hardware_connectivity.neighbors(node)):
                    #         nodes_to_remove.append(neighbor)
            
            # Remove nodes
            for node in set(nodes_to_remove):
                if hardware_connectivity.has_node(node):  # Double-check if the node exists in the graph
                    hardware_connectivity.remove_node(node)

            embeddings.update(emb)
        else:
            # If embedding fails, remove the corresponding Qubofrom the lists
            del QuboList[i]
            print("cannot embedded")
            
    return embeddings, embedding_list, TotalQubo, Qubo_list, offset_list, Qubits, Qubo_logic

from collections import defaultdict
from typing import Tuple

def auto_scale_qubo(qubo: defaultdict, 
                          h_range: float = 4.0, 
                          j_range: float = 1.0, 
                          j_extended_range: float = -2.0, 
                          per_qubit_coupling_range: Tuple[float, float] = (15, -18)):
    """
    Auto-scales a QUBO for the D-Wave Advantage 6.4, considering hardware constraints
    including the dynamic ranges for h and J coefficients and specific per-qubit coupling limits.

    Args:
        qubo: The input QUBO represented as a defaultdict.
        h_range: The dynamic range for h (linear) coefficients.
        j_range: The dynamic range for positive J (quadratic) coefficients.
        j_extended_range: The dynamic range for negative J (quadratic) coefficients.
        per_qubit_coupling_range: Tuple representing the dynamic range for total J per qubit,
                                  with positive and negative limits.

    Returns:
        A new auto-scaled QUBO as a defaultdict.
    """
    # Initialize for max and min checks
    max_h = max([value for (i, j), value in qubo.items() if i == j], default=0)
    min_h = min([value for (i, j), value in qubo.items() if i == j], default=0)
    max_J = max([value for (i, j), value in qubo.items() if i != j], default=0)
    min_J = min([value for (i, j), value in qubo.items() if i != j], default=0)

    total_J_per_qubit_pos = defaultdict(float)
    total_J_per_qubit_neg = defaultdict(float)
    for (i, j), value in qubo.items():
        if i != j:
            if value > 0:
                total_J_per_qubit_pos[i] += value
                total_J_per_qubit_pos[j] += value
            else:
                total_J_per_qubit_neg[i] += value
                total_J_per_qubit_neg[j] += value

    # Calculate coupling limit
    max_total_J = max(total_J_per_qubit_pos.values(), default=0)
    min_total_J = min(total_J_per_qubit_neg.values(), default=0)
    coupling_limit = max(
        max(max_total_J / max(per_qubit_coupling_range, default=0), 0),
        max(min_total_J / min(per_qubit_coupling_range, default=0), 0)
    )

    # Determine the scale factor
    scale_factor = max(
        max(max_h / h_range, 0),
        max(min_h / h_range, 0),
        max(max_J / j_range, 0),
        max(min_J / j_extended_range, 0),
        coupling_limit
    )

    # Prevent division by zero or negative scaling
    if scale_factor <= 0:
        scale_factor = 1

    # Apply the scaling factor
    scaled_qubo = defaultdict(float)
    for (i, j), value in qubo.items():
        scaled_qubo[(i, j)] = value / scale_factor

    return scaled_qubo

from dwave.embedding import unembed_sampleset
from dwave.embedding.chain_breaks import majority_vote as mv
from dwave.embedding.chain_breaks import weighted_random as wr
from dwave.embedding.chain_breaks import MinimizeEnergy as me

def unembed_solution(embedded_solution, embeddings, QUBO, chain_break_method):
    """
    Unembeds a solution from an embedded space back to the original space.

    Args:
        embedded_solution (dimod.SampleSet or dict): The solution to unembed. If it's not already a SampleSet, it will be converted to one.
        embeddings (dict): A dictionary mapping the variables in the embedded space to their corresponding variables in the original space.
        QUBO (dict): The QUBO matrix representing the problem.
        chain_break_method (str): The method to use for breaking chains during unembedding. Supported methods are 'majority_vote', 'weighted_random', and 'minimize_energy'.

    Returns:
        tuple: A tuple containing the unembedded solution and the time taken for unembedding.

    Raises:
        ValueError: If an unsupported chain break method is provided.

    """
    # Create the BinaryQuadraticModel from the QUBO.
    bqm = dimod.BinaryQuadraticModel.from_qubo(QUBO)

    # Convert the solution to a SampleSet if it's not already one.
    if not isinstance(embedded_solution, dimod.SampleSet):
        embedded_solution = dimod.SampleSet.from_samples(embedded_solution, energy=0, vartype=dimod.BINARY)
    
    # Determine the chain break method based on the provided argument.
    if chain_break_method == 'majority_vote':
        chain_break_method = mv
    elif chain_break_method == 'weighted_random':
        chain_break_method = wr
    elif chain_break_method == 'minimize_energy':
        chain_break_method = me(bqm, embeddings)
    else:
        raise ValueError(f"Unsupported chain break method: {chain_break_method}")

    start = time.time()
    
    # Unembed the solution.
    unembedded_solution = unembed_sampleset(embedded_solution, embeddings, bqm, chain_break_method=chain_break_method)

    unembed_t = time.time() - start

    return unembedded_solution, unembed_t

def calculate_energy(QUBO, solution, x, y):
    """
    Calculate the energy of a solution given a QUBO.

    Args:
        QUBO (dict): The QUBO dictionary where keys are tuples of variable indices, and values are the weights.
        solution (list): The solution for which to calculate the energy. It's a list of binary values (0s and 1s).
        x (int): The starting index of the variables in the solution list.
        y (int): The ending index of the variables in the solution list.

    Returns:
        float: The calculated energy of the solution.
    """
    energy = 0
    for (i, j), value in QUBO.items():
        if x <= i < y and x <= j < y:
            energy += value * solution[i - x] * solution[j - x]
    return energy

def qubo_logical(QUBO, x, y):
    """
    Extracts a sub-QUBO from the given QUBO dictionary based on the specified range of indices.

    Args:
        QUBO (dict): The input QUBO dictionary.
        x (int): The starting index (inclusive) of the sub-QUBO.
        y (int): The ending index (exclusive) of the sub-QUBO.

    Returns:
        dict: The sub-QUBO dictionary containing only the entries within the specified range.
    """
    qubo = {}
    for (i, j), value in QUBO.items():
        if x <= i < y and x <= j < y:
            qubo[(i, j)] = value
    return qubo

def response_decoder(response, x, y, QUBO_logic):
    """
    Decode the response from a quantum annealing solver and extract valid solutions.

    Args:
        response (dwave.cloud.Response): The response object containing the samples from the solver.
        x (int): The starting index of the solution segment in the response.
        y (int): The ending index of the solution segment in the response.
        QUBO_logic (dict): The QUBO logic dictionary.

    Returns:
        answer (list): A list of valid solutions, where each solution is represented as [solution_segment, energy].
        decode_time (float): The time taken for decoding the response.

    """
    start = time.time()
    answer = []

    # Iterate over the samples
    for indx, sample in enumerate(response.samples()):
        # Extract the solution segment corresponding to the current problem
        solution_segment = [sample[qubit] for qubit in range(x, y)]

        # Calculate the energy for the solution
        energy = calculate_energy(QUBO_logic, solution_segment, x, y)

        # Append the solution, and its energy to the list
        for _ in range(response.data_vectors['num_occurrences'][indx]):
            answer.append([solution_segment, energy])

    decode_time = time.time() - start
    return answer, decode_time

def unembed_combined_solution(embedded_solution, problem_embeddings, QUBO_logic, offsets, cmb):
    """
    Unembeds the solution from the quantum annealer's physical qubits back to the original problem's logical qubits.

    Args:
        embedded_solution (list of dimod.SampleSet): The solution obtained from the quantum annealer, represented in the embedded space.
        problem_embeddings (list): The embedding information used to map the original problem to the annealer's qubits.
        QUBO_logic (dict): The original QUBO dictionary.
        offsets (list): The offsets used to calculate the indices of the logical qubits in the QUBO.
        cmb (str): The method to resolve broken chains. Supported values are 'majority_vote', 'weighted_random', and 'minimize_energy'.

    Returns:
        tuple: The unembedded solution in the problem space of the original QUBO, along with the time taken for each unembedding operation.
    """

    unembedded_solution = []
    time_vectors = []
    for indx in range(len(problem_embeddings)):
        embeddings = problem_embeddings[indx]

        x, y = offsets[indx], offsets[indx+1]
        QUBO = qubo_logical(QUBO_logic, x, y)
        
        # Create the BinaryQuadraticModel from the QUBO.
        bqm = dimod.BinaryQuadraticModel.from_qubo(QUBO)

        # Convert the solution to a SampleSet if it's not already one.
        if not isinstance(embedded_solution, dimod.SampleSet):
            embedded_solution = dimod.SampleSet.from_samples(embedded_solution, energy=0, vartype=dimod.BINARY)
        
        # Determine the chain break method based on the provided argument.
        if cmb == 'majority_vote':
            chain_break_method = mv
        elif cmb == 'weighted_random':
            chain_break_method = wr
        elif cmb == 'minimize_energy':
            chain_break_method = me(bqm, embeddings)
        else:
            raise ValueError(f"Unsupported chain break method: {cmb}")

        start = time.time()
        
        # Unembed the solution.
        unembedded_solution.append(unembed_sampleset(embedded_solution, embeddings, bqm, chain_break_method=chain_break_method))

        time_vectors.append(time.time() - start)

    return unembedded_solution, np.average(time_vectors)
