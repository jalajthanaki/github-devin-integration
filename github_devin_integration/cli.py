"""CLI for GitHub Issues integration with Devin with async support."""
import os
from pathlib import Path
import asyncio
from typing import Optional, Dict, Any, List
import click
from github import Github, Auth
from github.Issue import Issue
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich import print as rprint
from dotenv import load_dotenv
import json
import aiohttp
import time


console = Console()

def load_env_safely() -> bool:
    """Load .env from user config directories, not package."""
    locations = [
        Path.cwd() / ".env",                              # Current directory
        Path.home() / ".github-devin" / ".env",           # User config dir
        Path.home() / ".config" / "github-devin" / ".env" # XDG standard
    ]

    for env_path in locations:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            console.print(f"[dim]Loaded environment from {env_path}[/dim]")
            return True

    return False

# Load environment on startup
if not load_env_safely():
    console.print("[yellow]No .env file found in default locations. please create a .env file with your GitHub token and devin api key at .~/.github-devin/.env or ~/.config/github-devin/.env or at current directory[/yellow]")
    console.print("Run [bold green]`github-devin setup`[/bold green] to configure.")


def get_github_client() -> Github:
    """Initialize and return GitHub client."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        console.print("[red]Error: GITHUB_TOKEN not found in .env file[/red]")
        console.print("Please create a .env file with your GitHub token.")
        console.print("See .env.example for reference.")
        raise click.Abort()
    return Github(auth=Auth.Token(token))


async def create_devin_session_async(
    issue_title: str,
    issue_description: str,
    issue_comments: list,
    devin_api_key: str
) -> Optional[str]:
    """Create a Devin session asynchronously and return the session ID."""
    try:
        comments_text = "\n\n".join([f"Comment: {comment}" for comment in issue_comments]) if issue_comments else "No comments"
        prompt = f"Scope the issue: {issue_title}\n\nDescription: {issue_description}\n\n{comments_text}\n\nScope the issue, provide key pointers to resolve the issue, and assign a confidence score (0-100%)."
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.devin.ai/v1/sessions",
                headers={"Authorization": f"Bearer {devin_api_key}"},
                json={"prompt": prompt, "idempotent": False},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                session_id = data.get("session_id")
                console.print(f"[green]Created Devin session: {session_id}[/green]")
                return session_id
                
    except aiohttp.ClientError as e:
        console.print(f"[red]Error creating Devin session: {str(e)}[/red]")
        return None


async def get_devin_session_async(session_id: str, devin_api_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve Devin session data asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.devin.ai/v1/sessions/{session_id}",
                headers={"Authorization": f"Bearer {devin_api_key}"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                return await response.json()
                
    except aiohttp.ClientError as e:
        console.print(f"[red]Error retrieving session: {str(e)}[/red]")
        return None


async def send_message_to_devin_session_async(
    session_id: str,
    devin_api_key: str,
    message: str
) -> Optional[Dict[str, Any]]:
    """Send a message to an existing Devin session asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.devin.ai/v1/sessions/{session_id}/message",
                headers={"Authorization": f"Bearer {devin_api_key}"},
                json={"message": message},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                console.print(f"[green]Message sent successfully[/green]")
                return result
                
    except aiohttp.ClientError as e:
        console.print(f"[red]Error sending message: {str(e)}[/red]")
        return None


async def monitor_devin_session_async(
    session_id: str,
    devin_api_key: str,
    max_wait_minutes: int = 10,
    callback=None,
    wait_for_new_response: bool = False
) -> Dict[str, Any]:
    """Monitor session status asynchronously with live updates.
    
    Args:
        session_id: The Devin session ID
        devin_api_key: API key for authentication
        max_wait_minutes: Maximum minutes to monitor
        callback: Optional callback function for status updates
        wait_for_new_response: If True, continue monitoring even in blocked state until new response
    """
    console.print(f"\n[yellow]Monitoring session {session_id}...[/yellow]")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    last_message_count = 0
    last_status = None
    initial_message_count = 0
    seen_message_ids = set()
    got_new_response_after_send = False
    
    # Get initial state if we're waiting for new response
    if wait_for_new_response:
        initial_data = await get_devin_session_async(session_id, devin_api_key)
        if initial_data:
            initial_message_count = len(initial_data.get("messages", []))
            # Track which messages we've already seen
            for msg in initial_data.get("messages", []):
                msg_id = f"{msg.get('timestamp', '')}-{msg.get('type', '')}"
                seen_message_ids.add(msg_id)
            console.print(f"[dim]Waiting for new responses (current message count: {initial_message_count})...[/dim]")
    
    poll_interval = 5  # Poll every 5 seconds for more responsive updates
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            console.print(f"\n[yellow]Timeout reached ({max_wait_minutes} minutes)[/yellow]")
            console.print("[dim]Session is still running. Check the web UI for updates.[/dim]")
            break
            
        status_data = await get_devin_session_async(session_id, devin_api_key)
        if not status_data:
            console.print("[red]Failed to fetch session data[/red]")
            await asyncio.sleep(poll_interval)
            continue
            
        status = status_data.get("status_enum", "unknown")
        messages = status_data.get("messages", [])
        current_message_count = len(messages)
        
        # Show status updates
        if status != last_status:
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            
            # Add more descriptive status messages
            status_emoji = {
                "running": "🔄",
                "blocked": "⏸️",
                "stopped": "⏹️",
                "completed": "✅"
            }
            emoji = status_emoji.get(status, "📊")
            
            console.print(f"[cyan]{emoji} Status: {status} (elapsed: {elapsed_min}m {elapsed_sec}s)[/cyan]")
            last_status = status
            
            # Call callback if provided
            if callback:
                await callback(status_data)
        
        # Show new messages - check each message to see if we've seen it
        if messages:
            for i, msg in enumerate(messages):
                msg_id = f"{msg.get('timestamp', '')}-{msg.get('type', '')}-{i}"
                
                if msg_id not in seen_message_ids:
                    seen_message_ids.add(msg_id)
                    msg_type = msg.get("type", "unknown")
                    content = msg.get("message", "")
                    
                    if msg_type == "devin_message":
                        console.print(f"\n[bold green]🤖 Devin:[/bold green]")
                        if len(content) > 4000:
                            console.print(content[:4000] + "...")
                            console.print(f"[dim](Message truncated - {len(content)} chars total)[/dim]")
                        else:
                            console.print(content)
                        
                        # Mark that we got a new response after initial state
                        if wait_for_new_response and i >= initial_message_count:
                            got_new_response_after_send = True
                            
                    elif msg_type == "user_message":
                        username = msg.get("username", "User")
                        console.print(f"\n[bold blue]👤 {username}:[/bold blue]")
                        if len(content) > 4000:
                            console.print(content[:4000] + "...")
                        else:
                            console.print(content)
        
        # Update last message count
        last_message_count = current_message_count
        
        # Check for terminal states
        if status in ["stopped", "completed"]:
            console.print(f"\n[green]✓ Session reached terminal state: {status}[/green]")
            
            # Show summary
            devin_messages = [m for m in messages if m.get("type") == "devin_message"]
            console.print(f"\n[bold]Session Summary:[/bold]")
            console.print(f"  Total messages: {len(messages)}")
            console.print(f"  Devin responses: {len(devin_messages)}")
            console.print(f"  Duration: {int(elapsed // 60)}m {int(elapsed % 60)}s")
            console.print(f"  Final status: {status}")
            
            return status_data
        
        # Handle blocked state
        if status == "blocked":
            if wait_for_new_response:
                # If we've already got a response, check if we should continue waiting
                if got_new_response_after_send:
                    # If we've been blocked for more than 2 minutes after getting a response, ask if we should continue
                    if elapsed > 300:  # 5 minutes
                        console.print("\n[yellow]Session has been blocked for 2+ minutes after receiving response.[/yellow]")
                        console.print("[yellow]Would you like to continue monitoring? (y/n)[/yellow] ")
                        
                        # Set up a way to check for user input asynchronously
                        try:
                            # Use asyncio.wait_for to add a timeout to the input
                            user_input = await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(
                                    None, 
                                    lambda: input().strip().lower()
                                ),
                                timeout=10.0  # 10 second timeout
                            )
                            
                            if user_input != 'y':
                                console.print("\n[yellow]Stopping monitoring as requested.[/yellow]")
                                return status_data
                                
                        except asyncio.TimeoutError:
                            console.print("\n[yellow]No response received, continuing to monitor...[/yellow]")
                        except Exception as e:
                            console.print(f"\n[yellow]Error getting user input: {e}[/yellow]")
                else:
                    # Still waiting for Devin to respond to our message
                    if int(elapsed) % 5 == 0:  # Only print a dot every 5 seconds
                        console.print(".", end="")
            else:
                # Not waiting for new response, stop at blocked state
                console.print(f"\n[yellow]Session is blocked (waiting for user input)[/yellow]")
                
                # Show summary
                devin_messages = [m for m in messages if m.get("type") == "devin_message"]
                console.print(f"\n[bold]Session Summary:[/bold]")
                console.print(f"  Total messages: {len(messages)}")
                console.print(f"  Devin responses: {len(devin_messages)}")
                console.print(f"  Duration: {int(elapsed // 60)}m {int(elapsed % 60)}s")
                console.print(f"  Current status: {status}")
                
                return status_data
        else:
            # Show progress indicator for non-blocked, non-terminal states
            console.print(".", end="")
        
        await asyncio.sleep(poll_interval)
    
    return await get_devin_session_async(session_id, devin_api_key) or {}


async def batch_analyze_issues(
    repo: str,
    issue_numbers: List[int],
    devin_api_key: str
) -> Dict[int, Dict[str, Any]]:
    """Analyze multiple issues concurrently."""
    g = get_github_client()
    repo_obj = g.get_repo(repo)
    
    console.print(f"\n[yellow]Analyzing {len(issue_numbers)} issues concurrently...[/yellow]\n")
    
    async def analyze_single_issue(issue_num: int) -> tuple[int, Optional[str]]:
        try:
            issue = repo_obj.get_issue(number=issue_num)
            comments = issue.get_comments()
            comment_bodies = [comment.body for comment in comments]
            
            console.print(f"[dim]Creating session for issue #{issue_num}...[/dim]")
            session_id = await create_devin_session_async(
                issue.title,
                issue.body or "",
                comment_bodies,
                devin_api_key
            )
            return issue_num, session_id
        except Exception as e:
            console.print(f"[red]Error analyzing issue #{issue_num}: {str(e)}[/red]")
            return issue_num, None
    
    # Create all sessions concurrently
    tasks = [analyze_single_issue(num) for num in issue_numbers]
    results = await asyncio.gather(*tasks)
    
    # Wait a bit for initial processing
    await asyncio.sleep(5)
    
    # Fetch all session data concurrently
    session_map = {num: sid for num, sid in results if sid}
    
    async def fetch_session_data(issue_num: int, session_id: str):
        data = await get_devin_session_async(session_id, devin_api_key)
        return issue_num, data
    
    session_tasks = [fetch_session_data(num, sid) for num, sid in session_map.items()]
    session_results = await asyncio.gather(*session_tasks)
    
    return {num: data for num, data in session_results if data}


def extract_confidence_score(session_data: Dict[str, Any]) -> Optional[float]:
    """Extract confidence score from Devin session messages."""
    messages = session_data.get("messages", [])
    
    for msg in reversed(messages):
        if msg.get("type") == "devin_message":
            message_text = msg.get("message", "").lower()
            
            import re
            patterns = [
                r'confidence score[:\s]+(\d+)%',
                r'confidence[:\s]+(\d+)%',
                r'confidence[:\s]+0?\.(\d+)',
                r'\*?\*?confidence\s*score\*?\*?[:\s\-]*([0-9]+(?:\.[0-9]+)?)\s*%?',  # e.g. **Confidence Score**: 95%
                r'\*?\*?confidence\*?\*?[:\s\-]*([0-9]+(?:\.[0-9]+)?)\s*%?',           # e.g. **Confidence**: 0.87 or 87%

            ]
            
            for pattern in patterns:
                match = re.search(pattern, message_text)
                if match:
                    score = float(match.group(1))
                    if score > 1:
                        return score / 100
                    return score
    
    return None


@click.group(invoke_without_command=True)
@click.option('--repo', help='Repository in format owner/repo')
@click.pass_context
def cli(ctx, repo):
    """GitHub Issues Integration for Devin with async support.
    
    Run 'github-devin setup' first to configure credentials.
    Then use commands like: github-devin --repo owner/repo list-issues
    """
    ctx.ensure_object(dict)
    
    # If no command was invoked and no repo specified, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()
    
    # Store repo if provided (some commands don't need it)
    if repo:
        ctx.obj['REPO'] = repo
    elif ctx.invoked_subcommand not in ['setup']:
        # Commands other than setup need a repo
        if ctx.invoked_subcommand:
            console.print("[red]Error: --repo is required for this command[/red]")
            console.print("Usage: github-devin --repo owner/repo <command>")
            ctx.exit(1)


@cli.command()
def setup():
    """Interactive setup for first-time configuration."""
    config_dir = Path.home() / ".github-devin"
    config_dir.mkdir(parents=True, exist_ok=True)
    env_file = config_dir / ".env"

    console.print("\n[bold blue]🔧 GitHub-Devin Setup[/bold blue]")
    console.print("Let's configure your credentials securely.\n")

    github_token = click.prompt("Enter your GitHub Token", hide_input=True)
    devin_key = click.prompt("Enter your Devin API Key", hide_input=True)

    with open(env_file, "w") as f:
        f.write(f"GITHUB_TOKEN={github_token}\n")
        f.write(f"DEVIN_API_KEY={devin_key}\n")

    if hasattr(os, "chmod"):
        os.chmod(env_file, 0o600)

    load_dotenv(env_file, override=True)
    console.print(f"\n[green]✅ Configuration saved at {env_file}[/green]")



@cli.command()
@click.option('--state', default='open', help='Filter by state (open, closed, all)')
@click.option('--label', help='Filter by label')
@click.option('--assignee', help='Filter by assignee')
@click.option('--limit', type=int, default=100, help='Limit number of issues to show')
@click.option('--json', 'as_json', is_flag=True, default=False, help='Output issues as JSON')
@click.pass_context
def list_issues(ctx, state: str, label: Optional[str], assignee: Optional[str], limit: int, as_json: bool):
    """List issues in the specified repository."""
    repo = ctx.obj['REPO']
    issue_list_json = []
    
    try:
        g = get_github_client()
        repo_obj = g.get_repo(repo)
        
        query_params = {'state': state}
        if label and label.lower() != "none":
            query_params['labels'] = [label]
        if assignee and assignee.lower() != "none":
            query_params['assignee'] = assignee
            
        issues = list(repo_obj.get_issues(**query_params)[:limit])
        
        if label and label.lower() == "none":
            issues = [issue for issue in issues if len(issue.labels) == 0]
        
        if assignee and assignee.lower() == "none":
            issues = [issue for issue in issues if len(issue.assignees) == 0]
        
        if as_json:
            for issue in issues:
                issue_list_json.append({
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "labels": [label.name for label in issue.labels],
                    "assignees": [assignee.login for assignee in issue.assignees] if issue.assignees else [],
                    "created_at": str(issue.created_at),
                    "url": issue.html_url
                })
            console.print(json.dumps(issue_list_json, indent=2))
        else:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim")
            table.add_column("Title")
            table.add_column("State")
            table.add_column("Labels")
            table.add_column("Assignee(s)")
            table.add_column("Created At")
            
            for issue in issues:
                labels = ", ".join([label.name for label in issue.labels])
                assignees = ", ".join([assignee.login for assignee in issue.assignees]) if issue.assignees else ""
                table.add_row(
                    str(issue.number),
                    f"[link={issue.html_url}]{issue.title}[/link]",
                    issue.state,
                    labels,
                    assignees,
                    str(issue.created_at),
                )
            
            console.print(f"\n[bold]Found {len(issues)} issue(s) for {repo_obj.full_name}[/bold]")
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@cli.command()
@click.argument('issue_number', type=int)
@click.option('--monitor', is_flag=True, help='Monitor the Devin session until completion')
@click.option('--session-id', help='Use existing Devin session ID instead of creating new one')
@click.pass_context
def analyze_issue(ctx, issue_number: int, monitor: bool, session_id: Optional[str]):
    """Analyze a specific issue in the repository and create/use a Devin session."""
    asyncio.run(_analyze_issue_async(ctx, issue_number, monitor, session_id))


async def _analyze_issue_async(ctx, issue_number: int, monitor: bool, session_id: Optional[str]):
    """Async implementation of analyze_issue."""
    repo = ctx.obj['REPO']
    
    try:
        g = get_github_client()
        repo_obj = g.get_repo(repo)
        issue = repo_obj.get_issue(number=issue_number)
        
        console.print(f"\n[bold]Issue #{issue_number}: {issue.title}[/bold]")
        console.print(f"[bold]URL:[/bold] {issue.html_url}")
        console.print(f"\n[bold]Description:[/bold]\n{issue.body}\n")
        
        comments = issue.get_comments()
        comment_bodies = []
        if comments.totalCount > 0:
            console.print(f"[bold]Comments ({comments.totalCount}):[/bold]")
            for comment in comments:
                console.print(f"\n{comment.user.login} at {comment.created_at}:")
                console.print(comment.body)
                comment_bodies.append(comment.body)
        else:
            console.print("[dim]No comments yet.[/dim]")
        
        devin_api_key = os.getenv("DEVIN_API_KEY")
        if not devin_api_key:
            console.print("[red]Error: DEVIN_API_KEY not found in .env file[/red]")
            return
        
        if session_id is not None and session_id.strip() != "":
            console.print(f"\n[yellow]Using existing session: {session_id}[/yellow]")
            session_data = await get_devin_session_async(session_id, devin_api_key)
        else:
            console.print("\n[yellow]Creating Devin session...[/yellow]")
            session_id = await create_devin_session_async(
                issue.title,
                issue.body or "",
                comment_bodies,
                devin_api_key
            )
            
            if not session_id:
                console.print("[red]Failed to create Devin session[/red]")
                return
            
            await asyncio.sleep(5)
            session_data = await get_devin_session_async(session_id, devin_api_key)
        
        if session_data:
            if monitor:
                session_data = await monitor_devin_session_async(session_id, devin_api_key)
            
            confidence_score = extract_confidence_score(session_data)
            
            console.print(f"\n[bold cyan]Session Status:[/bold cyan] {session_data.get('status_enum', 'unknown')}")
            
            if confidence_score is not None:
                console.print(f"[bold green]Confidence Score:[/bold green] {confidence_score:.0%}")
            else:
                console.print("[yellow]Confidence score not yet available (session may still be processing)[/yellow]")
            
            messages = session_data.get("messages", [])
            devin_messages = [m for m in messages if m.get("type") == "devin_message"]
            if devin_messages:
                latest = devin_messages[-1]
                console.print(f"\n[bold]Latest Devin Response:[/bold]")
                console.print(latest.get("message", "")) 
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


@cli.command()
@click.argument('issue_numbers', nargs=-1, type=int, required=True)
@click.option('--monitor', is_flag=True, help='Monitor all sessions until completion')
@click.pass_context
def batch_analyze(ctx, issue_numbers: tuple, monitor: bool):
    """Analyze multiple issues concurrently.
    
    Example: batch-analyze 31 32 33 --monitor
    """
    asyncio.run(_batch_analyze_async(ctx, list(issue_numbers), monitor))


async def _batch_analyze_async(ctx, issue_numbers: List[int], monitor: bool):
    """Async implementation of batch_analyze."""
    repo = ctx.obj['REPO']
    devin_api_key = os.getenv("DEVIN_API_KEY")
    
    if not devin_api_key:
        console.print("[red]Error: DEVIN_API_KEY not found in .env file[/red]")
        return
    
    # Analyze all issues concurrently
    results = await batch_analyze_issues(repo, issue_numbers, devin_api_key)
    
    # Display results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Issue #", style="dim")
    table.add_column("Status")
    table.add_column("Confidence", justify="right")
    table.add_column("Session ID")
    
    for issue_num, session_data in results.items():
        status = session_data.get("status_enum", "unknown")
        confidence = extract_confidence_score(session_data)
        session_id = session_data.get("session_id", "N/A")
        
        confidence_str = f"{confidence:.0%}" if confidence else "N/A"
        table.add_row(
            str(issue_num),
            status,
            confidence_str,
            session_id
        )
    
    console.print("\n[bold]Batch Analysis Results:[/bold]")
    console.print(table)
    
    # Monitor if requested
    if monitor and results:
        console.print("\n[yellow]Monitoring all sessions...[/yellow]")
        session_ids = [data.get("session_id") for data in results.values() if data.get("session_id")]
        
        tasks = [monitor_devin_session_async(sid, devin_api_key, max_wait_minutes=30) for sid in session_ids]
        await asyncio.gather(*tasks)


@cli.command()
@click.argument('issue_number', type=int)
@click.option('--session-id', required=True, help='Devin session ID to send action plan to')
@click.option('--monitor', is_flag=True, help='Monitor the session after sending message')
@click.option('--max-wait', type=int, default=30, help='Maximum minutes to monitor (default: 30)')
@click.pass_context
def resolve_issue(ctx, issue_number: int, session_id: str, monitor: bool, max_wait: int):
    """Send action plan message to existing Devin session to resolve the issue."""
    asyncio.run(_resolve_issue_async(ctx, issue_number, session_id, monitor, max_wait))


async def _resolve_issue_async(ctx, issue_number: int, session_id: str, monitor: bool, max_wait: int):
    """Async implementation of resolve_issue."""
    repo = ctx.obj['REPO']
    
    try:
        g = get_github_client()
        repo_obj = g.get_repo(repo)
        issue = repo_obj.get_issue(number=issue_number)
        
        console.print(f"\n[bold]Resolving Issue #{issue_number}: {issue.title}[/bold]")
        console.print(f"[dim]Repository: {repo}[/dim]")
        console.print(f"[dim]Session: https://app.devin.ai/sessions/{session_id.removeprefix("devin-")}[/dim]\n")
        
        devin_api_key = os.getenv("DEVIN_API_KEY")
        if not devin_api_key:
            console.print("[red]Error: DEVIN_API_KEY not found in .env file[/red]")
            return
        
        console.print("[yellow]Checking current session status...[/yellow]")
        current_status = await get_devin_session_async(session_id, devin_api_key)
        
        if current_status:
            status = current_status.get("status_enum", "unknown")
            console.print(f"[cyan]Current status: {status}[/cyan]")
            
            if status in ["stopped", "completed"]:
                console.print(f"[yellow]Session is already {status}. You may need to create a new session.[/yellow]")
                return
            
            if status == "blocked":
                console.print(f"[yellow]Session is blocked (waiting for input) - sending action plan will unblock it[/yellow]")
        
        message = """Based on the analysis above, please take the action plan, raise a PR if possible and complete the ticket. 

        Steps to follow:
        1. Review the root cause and proposed solutions
        2. Create a detailed implementation plan
        3. Execute the plan and make necessary code changes
        4. Test the solution thoroughly
        5. Provide a summary of changes made

        Please proceed with implementing the fix."""
        
        console.print(f"[yellow]Sending action plan to Devin...[/yellow]")
        result = await send_message_to_devin_session_async(session_id, devin_api_key, message)
        
        # Note: result might be None or empty dict on success, don't fail on that
        
        if monitor:
            console.print(f"\n[bold]Starting live monitoring (max {max_wait} minutes)...[/bold]")
            console.print("[dim]Tip: Press Ctrl+C to stop monitoring (session will continue running)[/dim]")
            
            try:
                # Clear any existing output before starting monitoring
                console.print("\n" + "-" * 80)
                
                # Wait a moment for Devin to start processing the message
                console.print("[dim]Waiting for Devin to start processing...[/dim]")
                await asyncio.sleep(5)
                
                # Start monitoring with wait_for_new_response=True to continue even if blocked
                session_data = await monitor_devin_session_async(
                    session_id, 
                    devin_api_key, 
                    max_wait_minutes=max_wait,
                    wait_for_new_response=True
                )
                
                if session_data:
                    console.print("\n" + "=" * 80)
                    console.print("[bold]Final Session Summary:[/bold]")
                    
                    messages = session_data.get("messages", [])
                    devin_messages = [m for m in messages if m.get("type") == "devin_message"]
                    
                    console.print(f"\nTotal messages: {len(messages)}")
                    console.print(f"Devin responses: {len(devin_messages)}")
                    console.print(f"Final status: {session_data.get('status_enum', 'unknown')}")
                    
                    if devin_messages:
                        console.print(f"\n[bold cyan]Latest Response:[/bold cyan]")
                        latest = devin_messages[-1]
                        response_text = latest.get("message", "")
                        if len(response_text) > 4000:
                            console.print(response_text[:4000] + "...")
                            console.print(f"[dim](Response truncated - {len(response_text)} chars total)[/dim]")
                        else:
                            console.print(response_text)
                    
                    pr_info = session_data.get("pull_request")
                    if pr_info:
                        console.print(f"\n[bold green]Pull Request Created![/bold green]")
                        console.print(f"PR URL: {pr_info}")
                    
                    console.print(f"\n[dim]Session URL: https://app.devin.ai/sessions/{session_id.removeprefix("devin-")}[/dim]")
                        
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Monitoring stopped by user[/yellow]")
                console.print(f"[dim]Session is still running at: https://app.devin.ai/sessions/{session_id.removeprefix("devin-")}[/dim]")
        else:
            console.print("\n[green]Action plan sent successfully![/green]")
            console.print(f"[dim]Monitor progress at: https://app.devin.ai/sessions/{session_id.removeprefix("devin-")}[/dim]")
            console.print(f"\n[dim]Tip: Use --monitor flag to watch progress in real-time[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@cli.command()
@click.argument('session_id')
@click.pass_context
def session_status(ctx, session_id: str):
    """Get status of a Devin session."""
    asyncio.run(_session_status_async(ctx, session_id))


async def _session_status_async(ctx, session_id: str):
    """Async implementation of session_status."""
    try:
        devin_api_key = os.getenv("DEVIN_API_KEY")
        if not devin_api_key:
            console.print("[red]Error: DEVIN_API_KEY not found in .env file[/red]")
            return
        
        console.print(f"\n[yellow]Fetching session {session_id}...[/yellow]")
        session_data = await get_devin_session_async(session_id, devin_api_key)
        
        if session_data:
            console.print(f"\n[bold]Session ID:[/bold] {session_data.get('session_id')}")
            console.print(f"[bold]Status:[/bold] {session_data.get('status_enum')}")
            console.print(f"[bold]Title:[/bold] {session_data.get('title')}")
            console.print(f"[bold]Created:[/bold] {session_data.get('created_at')}")
            console.print(f"[bold]Updated:[/bold] {session_data.get('updated_at')}")
            
            messages = session_data.get("messages", [])
            console.print(f"\n[bold]Messages ({len(messages)}):[/bold]")
            for msg in messages:
                msg_type = msg.get("type", "unknown")
                timestamp = msg.get("timestamp", "")
                content = msg.get("message", "")
                console.print(f"\n[cyan]{msg_type}[/cyan] at {timestamp}:")
                console.print(content[:4000] + ("..." if len(content) > 4000 else ""))
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


def main():
    try:
        cli(obj={})
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())