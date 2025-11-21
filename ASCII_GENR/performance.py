import time
import functools

class GlobalTimer:

    # Stores the execution time of each function
    timings = {}
    execution_count = {}
    global_start = None
    global_end = None

    @staticmethod
    def time(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the current time
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
            finally:
                # This runs even if the function raises an error
                end_time = time.perf_counter()
                duration = end_time - start_time

                # Add the current time spent on this function if its available
                if func.__name__ in GlobalTimer.timings:
                    current_time = GlobalTimer.timings[func.__name__]
                    duration = duration + current_time

                # Store the time in the class dictionary using the function name
                GlobalTimer.timings[func.__name__] = duration

                # Update counter
                if func.__name__ in GlobalTimer.execution_count:
                    count = GlobalTimer.execution_count[func.__name__]
                    count = count + 1
                    GlobalTimer.execution_count[func.__name__] = count
                else:
                    GlobalTimer.execution_count[func.__name__] = 1


            return result
        return wrapper

    @staticmethod
    def print_stats():
        # Get total time if available
        total_time = None
        if GlobalTimer.global_start is not None and GlobalTimer.global_end is not None:
            total_time = GlobalTimer.global_end - GlobalTimer.global_start


        # Prints all the stored timings
        print("\n" + "="*30)
        print("   PERFORMANCE METRICS")
        print("="*30)
        if not GlobalTimer.timings:
            print("No timings recorded.")
        else:
            for func_name, duration in GlobalTimer.timings.items():
                count = GlobalTimer.execution_count[func_name]
                if total_time is not None:
                    print(f"{func_name:<35} : {duration:.4f} sec total ({(duration / total_time) * 100:.2f}%) | execution count : {count}")
                else:
                    print(f"{func_name:<35} : {duration:.4f} seconds total | execution count : {count}")
        print("="*30 + "\n")

        if total_time is not None:
            print(f"TOTAL EXECUTION TIME: {total_time:.4f} seconds")

    @staticmethod
    def reset():
        GlobalTimer.timings.clear()
        GlobalTimer.execution_count.clear()
        GlobalTimer.global_start = None
        GlobalTimer.global_end = None

    @staticmethod
    def start():
        GlobalTimer.global_start = time.perf_counter()

    @staticmethod
    def end():
        GlobalTimer.global_end = time.perf_counter()
