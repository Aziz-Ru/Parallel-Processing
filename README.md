# Parallel-Processing

# High Performance Computing / Parallel Programming: Learning MPI

## 🔰 Table of Contents

- [What is MPI?](#what-is-mpi)
- [Why Use MPI?](#why-use-mpi)
- [MPI vs. Other Parallel Models](#mpi-vs-other-parallel-models)
- [MPI Environment Setup](#mpi-environment-setup)
- [Basic Concepts in MPI](#basic-concepts-in-mpi)
- [Hello World in MPI](#hello-world-in-mpi)
- [MPI Communicators](#mpi-communicators)
- [Point-to-Point Communication](#point-to-point-communication)
- [Collective Communication](#collective-communication)
- [MPI Data Types](#mpi-data-types)
- [Process Synchronization](#process-synchronization)
- [Parallel Algorithms with MPI](#parallel-algorithms-with-mpi)
- [Common Pitfalls & Debugging](#common-pitfalls--debugging)
- [Performance Tips](#performance-tips)
- [Advanced Topics](#advanced-topics)
- [Resources & Practice Problems](#resources--practice-problems)

---

## What is MPI?

Message Passing Interface (MPI) is a standardized and portable message-passing system designed for parallel programming in distributed memory architectures. It allows processes running on different nodes (computers or processors) to communicate and coordinate by sending and receiving messages. MPI is not a programming language but a library of functions that can be called from languages like C, C++, Fortran, or Python (via wrappers like mpi4py).

MPI was first developed in the early 1990s and has evolved through several versions. As of August 2025, the latest version is MPI 5.0, released on June 5, 2025, by the MPI Forum. It supports features like non-blocking communication, one-sided operations, and improved scalability for high-performance computing (HPC) environments.

Key features:
- **Portability**: Works across different hardware and operating systems.
- **Scalability**: Supports from a few processes to thousands on supercomputers.
- **Flexibility**: Handles point-to-point and collective communications.

MPI is widely used in scientific simulations, data analysis, and AI training on clusters.

---

## Why Use MPI?

MPI is essential for developing efficient parallel applications in HPC because:
- **Distributed Computing**: It enables programs to run on multiple machines, distributing workload and reducing computation time for large-scale problems (e.g., weather modeling or molecular dynamics).
- **Performance**: Optimized for low-latency communication, allowing fine-grained control over data transfer.
- **Standardization**: As a de facto standard, MPI code is portable across clusters, supercomputers, and cloud environments.
- **Community and Support**: Backed by implementations like OpenMPI, MPICH, and Intel MPI, with extensive libraries and tools.
- **Handling Big Data**: Ideal for problems too large for single-node memory, like matrix operations or simulations.

Without MPI, parallel programming in distributed systems would require custom communication protocols, leading to inefficiency and non-portability.

---

## MPI vs. Other Parallel Models

MPI focuses on distributed memory parallelism via message passing. Here's a comparison with other models:

- **MPI (Message Passing Interface)**:
  - **Model**: Distributed memory; processes communicate explicitly via messages.
  - **Strengths**: Scalable to thousands of nodes, portable, supports heterogeneous systems.
  - **Weaknesses**: Explicit communication can be complex; higher overhead for small messages.
  - **Use Cases**: Cluster-based HPC, like simulations on supercomputers.

- **OpenMP (Open Multi-Processing)**:
  - **Model**: Shared memory; uses compiler directives for threading on multi-core CPUs.
  - **Strengths**: Easy to implement (add pragmas to loops), low overhead, automatic data sharing.
  - **Weaknesses**: Limited to single node (no distributed memory), not scalable beyond one machine.
  - **Use Cases**: Multi-core CPU parallelism, e.g., loop parallelization in scientific codes.

- **CUDA (Compute Unified Device Architecture)**:
  - **Model**: GPU parallelism; kernel launches for massive threading on NVIDIA GPUs.
  - **Strengths**: Extremely high throughput for data-parallel tasks (e.g., matrix multiplication), leverages GPU's thousands of cores.
  - **Weaknesses**: Hardware-specific (NVIDIA only), requires data transfer between CPU/GPU, not for distributed systems.
  - **Use Cases**: AI training, graphics, simulations on GPUs.

**Comparison Summary**:
| Aspect              | MPI                  | OpenMP               | CUDA                 |
|---------------------|----------------------|----------------------|----------------------|
| Memory Model       | Distributed         | Shared              | Shared (GPU)        |
| Scalability        | High (clusters)     | Medium (multi-core) | High (single GPU)   |
| Complexity         | High (explicit comm)| Low (directives)    | Medium (kernels)    |
| Hardware           | CPUs, clusters      | Multi-core CPUs     | NVIDIA GPUs         |

Hybrid approaches (e.g., MPI + OpenMP for distributed multi-core, MPI + CUDA for GPU clusters) are common for optimal performance.

---

## MPI Environment Setup

Setting up MPI varies by OS. Common implementations: OpenMPI (open-source), MPICH (portable), Intel MPI (optimized for Intel hardware).

### On Linux (e.g., Ubuntu)
1. Install OpenMPI:
   ```
   sudo apt update
   sudo apt install openmpi-bin openmpi-common libopenmpi-dev
   ```
2. Verify:
   ```
   mpicc --version
   mpirun --version
   ```

### On Windows
1. Download Microsoft MPI (MS-MPI) from Microsoft's site.
2. Install the SDK and runtime.
3. Add to PATH: `C:\Program Files\Microsoft MPI\Bin`.
4. Verify:
   ```
   mpicc --version
   ```

### On MacOS
1. Install via Homebrew:
   ```
   brew install open-mpi
   ```
2. Verify as above.

### For Development
- Use a C/C++/Fortran compiler (e.g., gcc, gfortran).
- Compile MPI code: `mpicc -o program program.c`
- Run: `mpirun -np 4 ./program` (for 4 processes).

Note: For Python, install mpi4py: `pip install mpi4py` (requires MPI installed).

---

## Basic Concepts in MPI

- **Process**: Independent execution unit with its own memory space.
- **Rank**: Unique ID for each process (0 to size-1).
- **Communicator**: Group of processes that can communicate (e.g., MPI_COMM_WORLD for all processes).
- **Message**: Data sent between processes, including tag, source/destination rank, and data type.
- **Blocking vs. Non-Blocking**: Blocking waits for completion; non-blocking allows overlap with computation.
- **Collective Operations**: Involve all processes in a communicator (e.g., broadcast).
- **Point-to-Point**: Between two specific processes (send/receive).

MPI programs start with `MPI_Init` and end with `MPI_Finalize`.

---

## Hello World in MPI

A basic MPI program prints "Hello World" from each process, showing its rank.

**C Example** (hello.c):
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get number of processes

    printf("Hello World from process %d of %d\n", rank, size);

    MPI_Finalize();  // Finalize MPI
    return 0;
}
```

Compile: `mpicc -o hello hello.c`

Run: `mpirun -np 4 ./hello`

Output (example):
```
Hello World from process 0 of 4
Hello World from process 1 of 4
Hello World from process 2 of 4
Hello World from process 3 of 4
```

This demonstrates basic initialization, rank querying, and collective termination.

---

## MPI Communicators

Communicators define groups of processes for communication.
- **MPI_COMM_WORLD**: Default communicator including all processes.
- **Creating Custom Communicators**:
  - `MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);` – Splits based on color (group ID).
- **Example**: Split into even/odd ranks.
  ```c
  int color = rank % 2;
  MPI_Comm new_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
  ```
- **Use**: For sub-groups, e.g., row/column communicators in matrix operations.

Communicators ensure messages are scoped, preventing interference.

---

## Point-to-Point Communication

Direct communication between two processes.
- **Blocking Send/Receive**:
  - `MPI_Send(buf, count, datatype, dest, tag, comm);`
  - `MPI_Recv(buf, count, datatype, source, tag, comm, &status);`
- **Example**: Process 0 sends data to process 1.
  ```c
  if (rank == 0) {
      int data = 42;
      MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (rank == 1) {
      int data;
      MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("Received %d\n", data);
  }
  ```
- **Non-Blocking**: `MPI_Isend`, `MPI_Irecv` for overlap (wait with `MPI_Wait`).

Used for tasks like data exchange in simulations.

---

## Collective Communication

Operations involving all processes in a communicator.
- **Broadcast**: `MPI_Bcast(buf, count, datatype, root, comm);` – Root sends to all.
- **Scatter**: `MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);` – Distributes data.
- **Gather**: `MPI_Gather` – Collects data to root.
- **Reduce**: `MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);` – e.g., sum with `MPI_SUM`.
- **Example (Broadcast)**:
  ```c
  int data = (rank == 0) ? 42 : 0;
  MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("Process %d received %d\n", rank, data);
  ```

Efficient for global operations, synchronized.

---

## MPI Data Types

MPI defines types for safe data transfer.
- **Basic**: `MPI_INT`, `MPI_FLOAT`, `MPI_CHAR`, etc.
- **Derived**: Custom types for structs/arrays.
  - `MPI_Type_contiguous(count, oldtype, &newtype);` – Contiguous array.
  - `MPI_Type_vector(count, blocklength, stride, oldtype, &newtype);` – Strided array.
  - Commit with `MPI_Type_commit(&newtype);`.
- **Example (Derived Vector)**:
  ```c
  MPI_Datatype vec_type;
  MPI_Type_vector(3, 2, 4, MPI_INT, &vec_type);
  MPI_Type_commit(&vec_type);
  MPI_Send(buf, 1, vec_type, dest, tag, comm);
  ```

Ensures portability across architectures.

---

## Process Synchronization

Ensures processes coordinate timing.
- **Barrier**: `MPI_Barrier(comm);` – All wait until everyone reaches it.
- **Synchronization in Non-Blocking**: Use `MPI_Wait` or `MPI_Test` on requests.
- **Example**:
  ```c
  // Computation...
  MPI_Barrier(MPI_COMM_WORLD);
  // All processes synchronized
  ```
- **Advanced**: Fences in one-sided communication.

Prevents race conditions, ensures data consistency.

---

## Parallel Algorithms with MPI

MPI enables parallelizing algorithms like matrix multiplication.

**Example: Matrix Multiplication (A * B = C)**:
- Distribute rows of A to processes; broadcast B.
- Each process computes its portion of C.
- Gather results at root.

**C Code Snippet** (Simplified):
```c
// Assume matrices are square, size N
if (rank == 0) {
    // Initialize A, B
}
MPI_Scatter(A, N*N/num_procs, MPI_FLOAT, local_A, N*N/num_procs, MPI_FLOAT, 0, comm);
MPI_Bcast(B, N*N, MPI_FLOAT, 0, comm);
// Local multiply: local_C = local_A * B
MPI_Gather(local_C, N*N/num_procs, MPI_FLOAT, C, N*N/num_procs, MPI_FLOAT, 0, comm);
```

Other algorithms: Parallel sorting (e.g., odd-even sort), graph traversal (e.g., BFS with message passing).

---

## Common Pitfalls & Debugging

**Pitfalls**:
- **Deadlocks**: Mismatched send/receive (e.g., all send without recv).
- **Buffer Overflows**: Sending more data than expected.
- **Rank Errors**: Incorrect source/dest ranks.
- **Non-Portable Types**: Use MPI datatypes instead of sizeof.
- **Scalability Issues**: Poor load balancing.

**Debugging Tips**:
- Use `printf` with rank: `printf("Rank %d: ...\n", rank);`
- Tools: Valgrind for memory leaks, GDB with MPI (e.g., `mpirun -np 4 gdb ./program`).
- MPI-specific: MPICH's `--verbose`, or tools like MUST for correctness checking.
- Performance: Use MPI profiling interfaces (e.g., MPI_Pcontrol).

Avoid common errors by matching tags and communicators.

---

## Performance Tips

- **Overlap Communication/Computation**: Use non-blocking calls (Isend/Irecv) and MPI_Wait.
- **Minimize Synchronization**: Avoid unnecessary barriers.
- **Optimize Data Transfer**: Use derived datatypes for non-contiguous data.
- **Load Balancing**: Distribute work evenly.
- **Tuning**: Use MPI environment variables (e.g., `OMPI_MCA_btl_tcp_if_include=eth0` for network).
- **Collectives**: Prefer optimized collectives over manual point-to-point.
- **Profiling**: Use tools like TAU or Scalasca to identify bottlenecks.

Benchmark with varying process counts.

---

## Advanced Topics

- **Non-Blocking Communication**: `MPI_Isend`, `MPI_Irecv` for asynchronous ops, allowing computation overlap.
- **Persistent Communication**: `MPI_Send_init`, `MPI_Recv_init` for repeated patterns, reducing overhead.
- **One-Sided Communication (RMA)**: `MPI_Put`, `MPI_Get` for direct memory access without receiver involvement.
- **Dynamic Processes**: `MPI_Comm_spawn` to launch processes at runtime.
- **Fault Tolerance**: MPI-4+ features like sessions for resilience.
- **Hybrid MPI + Threads**: Combine with OpenMP for multi-core nodes.

Example (Non-Blocking):
```c
MPI_Request req;
MPI_Isend(buf, count, datatype, dest, tag, comm, &req);
// Computation...
MPI_Wait(&req, &status);
```

These enable efficient large-scale apps.

---

## Resources & Practice Problems

**Resources**:
- **Official**: MPI Forum (mpi-forum.org) – Standards docs.
- **Tutorials**: mpitutorial.com (basic to advanced), LLNL MPI Tutorial (hpc-tutorials.llnl.gov/mpi), Princeton Resources (researchcomputing.princeton.edu/education/external-online-resources/mpi).
- **Books**: "Using MPI" by Gropp et al., "Parallel Programming with MPI" by Pacheco.
- **Implementations**: OpenMPI (open-mpi.org), MPICH (mpich.org).
- **Tools**: mpi4py for Python, Valgrind/GDB for debugging.

**Practice Problems**:
1. **Hello World Variant**: Modify to print ranks in order (use barriers).
2. **Ping-Pong**: Measure latency with point-to-point sends/recvs.
3. **Matrix Multiplication**: Implement distributed version (as above).
4. **Pi Calculation**: Use Monte Carlo method with MPI_Reduce.
5. **Image Processing**: Parallelize convolution (e.g., edge detection) across processes.
6. **Advanced**: Implement non-blocking matrix multiply with overlap.

Exercises from RookieHPC (rookiehpc.org/mpi/exercises) or LLNL (hpc-tutorials.llnl.gov/mpi/exercise_1). Practice on a cluster or local machine with multiple cores.

