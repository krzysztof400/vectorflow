import numpy as np
import vectorflow

def run_tests():
    """
    Runs tests for the vectorflow library.
    """
    print("--- Starting vectorflow library tests ---")

    # Test data
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    B = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
    
    np_A = np.array(A)
    np_B = np.array(B)

    print("\nMatrix A:\n", np_A)
    print("Matrix B:\n", np_B)

    # --- Test Matrix Addition ---
    print("\n--- Testing Matrix Addition ---")
    try:
        result_add = vectorflow.add_matrices(A, B)
        expected_add = np_A + np_B
        print("vectorflow.add_matrices result:\n", np.array(result_add))
        print("Numpy expected result:\n", expected_add)
        assert np.allclose(result_add, expected_add)
        print("✅ Addition Test Passed")
    except Exception as e:
        print(f"❌ Addition Test Failed: {e}")

    # --- Test Matrix Multiplication ---
    print("\n--- Testing Matrix Multiplication ---")
    try:
        result_mul = vectorflow.multiply_matrices(A, B)
        expected_mul = np_A @ np_B
        print("vectorflow.multiply_matrices result:\n", np.array(result_mul))
        print("Numpy expected result:\n", expected_mul)
        assert np.allclose(result_mul, expected_mul)
        print("✅ Multiplication Test Passed")
    except Exception as e:
        print(f"❌ Multiplication Test Failed: {e}")

    # --- Test Model ---
    print("\n--- Testing Model ---")
    try:
        model = vectorflow.Model()
        # Assuming a train method exists as per README
        # You might need to adjust this based on your model's actual API
        dummy_data = [[1.1, 2.2], [3.3, 4.4]]
        print("Instantiating and training model with dummy data...")
        model.train(dummy_data)
        print("✅ Model Test Passed (instantiation and train call)")
    except AttributeError:
         print("⚠️  Model or train method not found in bindings. Skipping test.")
    except Exception as e:
        print(f"❌ Model Test Failed: {e}")

    print("\n--- All tests finished ---")


if __name__ == "__main__":
    # First, you need to build and install the library.
    # From your terminal in the 'vectorflow-python' directory, run:
    # pip install .
    #
    # After installation, you can run this test script:
    # python test.py
    
    try:
        run_tests()
    except ImportError:
        print("Could not import 'vectorflow'.")
        print("Please make sure you have installed the library by running 'pip install .' in the 'vectorflow-python' directory.")



# To use this script:

# 1.  Make sure you have `numpy` installed (`pip install numpy`).
# 2.  Build and install your `vectorflow` library by running `pip install .` inside the `/home/krzysztof/Dev/vectorflow/vectorflow-python` directory.
# 3.  Run the test script with `python test.py`.// filepath: /home/krzysztof/Dev/vectorflow/vectorflow-python/test.py

import numpy as np
import vectorflow

def run_tests():
    """
    Runs tests for the vectorflow library.
    """
    print("--- Starting vectorflow library tests ---")

    # Test data
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    B = [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]
    
    np_A = np.array(A)
    np_B = np.array(B)

    print("\nMatrix A:\n", np_A)
    print("Matrix B:\n", np_B)

    # --- Test Matrix Addition ---
    print("\n--- Testing Matrix Addition ---")
    try:
        result_add = vectorflow.add_matrices(A, B)
        expected_add = np_A + np_B
        print("vectorflow.add_matrices result:\n", np.array(result_add))
        print("Numpy expected result:\n", expected_add)
        assert np.allclose(result_add, expected_add)
        print("✅ Addition Test Passed")
    except Exception as e:
        print(f"❌ Addition Test Failed: {e}")

    # --- Test Matrix Multiplication ---
    print("\n--- Testing Matrix Multiplication ---")
    try:
        result_mul = vectorflow.multiply_matrices(A, B)
        expected_mul = np_A @ np_B
        print("vectorflow.multiply_matrices result:\n", np.array(result_mul))
        print("Numpy expected result:\n", expected_mul)
        assert np.allclose(result_mul, expected_mul)
        print("✅ Multiplication Test Passed")
    except Exception as e:
        print(f"❌ Multiplication Test Failed: {e}")

    # --- Test Model ---
    print("\n--- Testing Model ---")
    try:
        model = vectorflow.Model()
        # Assuming a train method exists as per README
        # You might need to adjust this based on your model's actual API
        dummy_data = [[1.1, 2.2], [3.3, 4.4]]
        print("Instantiating and training model with dummy data...")
        model.train(dummy_data)
        print("✅ Model Test Passed (instantiation and train call)")
    except AttributeError:
         print("⚠️  Model or train method not found in bindings. Skipping test.")
    except Exception as e:
        print(f"❌ Model Test Failed: {e}")

    print("\n--- All tests finished ---")


if __name__ == "__main__":
    # First, you need to build and install the library.
    # From your terminal in the 'vectorflow-python' directory, run:
    # pip install .
    #
    # After installation, you can run this test script:
    # python test.py
    
    try:
        run_tests()
    except ImportError:
        print("Could not import 'vectorflow'.")
        print("Please make sure you have installed the library by running 'pip install .' in the 'vectorflow-python' directory.")

# ```

# To use this script:

# 1.  Make sure you have `numpy` installed (`pip install numpy`).
# 2.  Build and install your `vectorflow` library by running `pip install .` inside the `/home/krzysztof/Dev/vectorflow/vectorflow-python` directory.
# 3.  Run the test script with `python test.py`.