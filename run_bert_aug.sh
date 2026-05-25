echo "Running BERT model comparisons for Task 1, Task 2, and Task 3..."
cd src/task1/
uv run bert_based_comparison.py --augment
cd ../task2/
echo "Task 1 completed. Moving on to Task 2..."
uv run bert_based_comparison.py --augment
cd ../task3/
echo "Task 2 completed. Moving on to Task 3..."
uv run bert_based_comparison.py --augment
cd ../../
echo "All BERT model comparisons completed successfully!"
