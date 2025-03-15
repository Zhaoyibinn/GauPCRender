from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

def get_progress():
    progress = Progress(TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                TextColumn("[red] {task.fields[loss]}"), 
                )
    return progress

progress = get_progress()