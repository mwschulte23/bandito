"""
Bandito Python SDK

A lightweight client for the Bandito multi-armed bandit API.

Usage:
    from bandito import BanditoClient

    client = BanditoClient("http://localhost:8000")
    client.login("user@example.com", "password")

    # Pull an arm
    result = client.pull(bandit_id=1, query="What is machine learning?")

    # Call your LLM with result.arm...

    # Submit reward
    client.reward(
        bandit_id=1,
        event_id=result.event_id,
        score=0.85,
        llm_output={"response": "..."},
        cost=0.02,
        latency=350
    )

    # View performance
    lb = client.leaderboard(bandit_id=1)
    analysis = client.analysis(bandit_id=1)

    # Check budget
    budget = client.budget(bandit_id=1)
    if result.budget_warning:
        print(f"Warning: {result.budget_warning}")
"""

import httpx
from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel


class BanditoError(Exception):
    """Base exception for Bandito SDK errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(BanditoError):
    """Raised when authentication fails."""
    pass


class NotFoundError(BanditoError):
    """Raised when a resource is not found."""
    pass


class Arm(BaseModel):
    """Represents a bandit arm configuration."""
    id: int
    bandit_id: int
    model_name: str
    system_prompt: str
    arm_metadata: List[Any] = []
    is_active: bool = True


class PullResult(BaseModel):
    """Result from pulling an arm."""
    event_id: int
    arm: Arm
    context: Dict[str, Any]
    budget_warning: Optional[str] = None


class Event(BaseModel):
    """Represents a bandit event for human review."""
    id: int
    bandit_id: int
    arm_id: int
    context: Dict[str, Any]
    model_score: float
    user_query: str
    llm_output: Optional[Dict[str, Any]] = None
    immediate_reward: Optional[float] = None
    human_reward: Optional[float] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    created_at: str
    updated_at: str


class ArmStats(BaseModel):
    """Stats for a single arm in the leaderboard."""
    arm_id: int
    model_name: str
    system_prompt: str
    is_active: bool
    pull_count: int
    avg_immediate_reward: float
    avg_human_reward: Optional[float] = None
    # Raw values
    avg_cost: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    total_cost: Optional[float] = None
    total_latency_ms: Optional[float] = None
    # Human-friendly display
    avg_cost_display: Optional[str] = None
    total_cost_display: Optional[str] = None
    avg_latency_display: Optional[str] = None
    total_latency_display: Optional[str] = None


class Leaderboard(BaseModel):
    """Leaderboard results for a bandit."""
    bandit_id: int
    total_pulls: int
    arms: List[ArmStats]


class Budget(BaseModel):
    """Budget status for a bandit."""
    bandit_id: int
    budget: Optional[float]
    current_spend: float
    budget_remaining: Optional[float]
    budget_used_percent: Optional[float]
    is_over_budget: bool


class BanditoClient:
    """
    Client for interacting with the Bandito API.

    Args:
        base_url: The base URL of the Bandito API (e.g., "http://localhost:8000")
        timeout: Request timeout in seconds (default: 30)
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v1"
        self._token: Optional[str] = None
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        """Build request headers with auth token."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an authenticated request to the API."""
        url = f"{self.api_url}{endpoint}"

        response = self._client.request(
            method=method,
            url=url,
            headers=self._headers(),
            json=json,
            params=params
        )

        if response.status_code == 401:
            raise AuthenticationError("Authentication failed. Please login.", 401)
        if response.status_code == 404:
            raise NotFoundError(f"Resource not found: {endpoint}", 404)
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            raise BanditoError(f"API error: {detail}", response.status_code)

        return response.json()

    # ============ Auth ============

    def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate with the API.

        Args:
            email: User email
            password: User password

        Returns:
            User info dict with id, email, is_superuser

        Raises:
            AuthenticationError: If credentials are invalid
        """
        response = self._client.post(
            f"{self.api_url}/auth/login",
            data={"username": email, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if response.status_code != 200:
            raise AuthenticationError("Invalid credentials", response.status_code)

        data = response.json()
        self._token = data["access_token"]
        return data.get("user", {})

    def set_token(self, token: str) -> None:
        """
        Set the auth token directly (useful if you already have one).

        Args:
            token: JWT access token
        """
        self._token = token

    @property
    def is_authenticated(self) -> bool:
        """Check if client has an auth token set."""
        return self._token is not None

    # ============ Core Actions ============

    def pull(
        self,
        bandit_id: int,
        query: str,
        gamma: float = 1.0,
        beta: float = 1.0
    ) -> PullResult:
        """
        Pull an arm using Thompson Sampling.

        Args:
            bandit_id: ID of the bandit to pull from
            query: The user query being processed
            gamma: Regularization parameter (default: 1.0)
            beta: Exploration parameter (default: 1.0)

        Returns:
            PullResult with event_id, selected arm, and context

        Example:
            result = client.pull(1, "Explain quantum computing")
            print(f"Use model: {result.arm.model_name}")
            print(f"System prompt: {result.arm.system_prompt}")
        """
        data = self._request("POST", f"/bandit/{bandit_id}/pull", json={
            "user_query": query,
            "gamma": gamma,
            "beta": beta
        })

        return PullResult(
            event_id=data["event_id"],
            arm=Arm.model_validate(data["arm"]),
            context=data["context"],
            budget_warning=data.get("budget_warning")
        )

    def reward(
        self,
        bandit_id: int,
        event_id: int,
        score: float,
        llm_output: str | Dict,
        cost: float,
        latency: float
    ) -> Dict[str, Any]:
        """
        Submit immediate reward after LLM response.

        Args:
            bandit_id: ID of the bandit
            event_id: ID of the event (from pull result)
            score: Reward score (typically 0-1)
            llm_output: LLM response (stored for human review)
            cost: API cost in dollars
            latency: Response time in milliseconds

        Returns:
            Updated event data

        Example:
            client.reward(
                bandit_id=1,
                event_id=42,
                score=0.9,
                llm_output={"response": "Quantum computing uses..."},
                cost=0.02,
                latency=450
            )
        """
        return self._request(
            "POST",
            f"/bandit/{bandit_id}/events/{event_id}/reward/immediate",
            json={
                "score": score,
                "llm_output": llm_output,
                "cost": cost,
                "latency": latency
            }
        )

    def feedback(
        self,
        bandit_id: int,
        event_id: int,
        score: Literal[0, 1]
    ) -> Dict[str, Any]:
        """
        Submit human feedback for an event.

        Args:
            bandit_id: ID of the bandit
            event_id: ID of the event
            score: Human rating (0 = bad, 1 = good)

        Returns:
            Updated event data

        Example:
            client.feedback(bandit_id=1, event_id=42, score=1)
        """
        return self._request(
            "POST",
            f"/bandit/{bandit_id}/events/{event_id}/reward/human",
            json={"score": score}
        )

    def get_event(self, bandit_id: int, event_id: int) -> Event:
        """
        Get an event for human review.

        Args:
            bandit_id: ID of the bandit
            event_id: ID of the event

        Returns:
            Event with full details including llm_output and user_query

        Example:
            event = client.get_event(bandit_id=1, event_id=42)
            print(f"Query: {event.user_query}")
            print(f"Response: {event.llm_output}")
            # Review and submit feedback
            client.feedback(bandit_id=1, event_id=42, score=1)
        """
        data = self._request("GET", f"/bandit/{bandit_id}/events/{event_id}")
        return Event.model_validate(data)

    # ============ Analysis ============

    def leaderboard(self, bandit_id: int) -> Leaderboard:
        """
        Get the event-based leaderboard for a bandit.

        Args:
            bandit_id: ID of the bandit

        Returns:
            Leaderboard with per-arm stats sorted by avg reward

        Example:
            lb = client.leaderboard(1)
            for arm in lb.arms:
                print(f"{arm.model_name}: {arm.avg_immediate_reward:.2f} ({arm.pull_count} pulls)")
        """
        data = self._request("GET", f"/bandit/{bandit_id}/leaderboard")
        return Leaderboard.model_validate(data)

    def analysis(self, bandit_id: int) -> Dict[str, Any]:
        """
        Get state-based analysis for a bandit.

        Returns decomposed theta_hat weights showing:
        - Model baseline effects
        - Prompt effects
        - Time interaction weights
        - Uncertainty estimates

        Args:
            bandit_id: ID of the bandit

        Returns:
            Analysis dict with models, prompts, and summary

        Example:
            analysis = client.analysis(1)
            for model, data in analysis["models"].items():
                print(f"{model}: baseline={data['baseline_effect']:.3f}")
        """
        return self._request("GET", f"/bandit/{bandit_id}/analysis")

    def budget(self, bandit_id: int) -> Budget:
        """
        Get budget status for a bandit.

        Args:
            bandit_id: ID of the bandit

        Returns:
            Budget with current spend, limit, and status

        Example:
            budget = client.budget(1)
            print(f"Spent ${budget.current_spend:.2f} of ${budget.budget:.2f}")
            if budget.budget_used_percent and budget.budget_used_percent >= 90:
                print("Warning: approaching budget limit!")
        """
        data = self._request("GET", f"/bandit/{bandit_id}/budget")
        return Budget.model_validate(data)

    # ============ Cleanup ============

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
