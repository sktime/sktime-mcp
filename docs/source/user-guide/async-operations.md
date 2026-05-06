# Asynchronous Operations

For tasks that require significant time or computing power, you can ask your AI assistant to handle them in the background. This keeps your conversation responsive while the heavy work is performed.

## When to Use Background Tasks

Consider asking for background execution when:
- **Loading large datasets**: If you are ingesting millions of rows from a database or a file.
- **Training complex models**: If you are using deep learning or ensemble methods that take minutes to fit.
- **Extensive evaluation**: If you are running cross-validation with many folds.

## How to Start a Background Task

Simply mention to your assistant that you want the operation to run in the background.

> *Example: "Fit this Prophet model in the background on my sales data."*

The assistant will initiate the job and give you a confirmation. It will also mention a `job_id` (e.g., `job_8e1f2a`), which the system uses to track progress.

## Monitoring Progress

You don't need to wait silently. You can ask your assistant for updates at any time.

> *Example: "What is the status of my background training job?"*

The assistant will report the current state:
- **Pending/Running**: The job is still in progress.
- **Completed**: The job is finished, and the results (like a model handle or a forecast) are ready for use.
- **Failed**: Something went wrong, and the assistant will provide the error details.

## Managing Jobs

If you change your mind, you can tell the assistant to cancel a task.

> *Example: "Cancel that background training job."*

The assistant will terminate the operation and free up the associated system resources.
