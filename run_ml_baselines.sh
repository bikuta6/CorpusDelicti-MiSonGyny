echo "Running ML baselines for Task 1, Task 2, and Task 3..."
uv run src/task1/ml_baseline.py
echo "Task 1 completed. Moving on to Task 2..."
uv run src/task2/ml_baseline.py
echo "Task 2 completed. Moving on to Task 3..."
uv run src/task3/ml_baseline.py
echo "All ML baselines completed successfully!"
