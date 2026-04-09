echo "Running ML comparisons for Task 1, Task 2, and Task 3..."
uv run src/task1/ml_comparison.py --baseline
uv run src/task1/ml_comparison.py
echo "Task 1 completed. Moving on to Task 2..."
uv run src/task2/ml_comparison.py --baseline
uv run src/task2/ml_comparison.py
echo "Task 2 completed. Moving on to Task 3..."
uv run src/task3/ml_comparison.py --baseline
uv run src/task3/ml_comparison.py
echo "All ML comparisons completed successfully!"
