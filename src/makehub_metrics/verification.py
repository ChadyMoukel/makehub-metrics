#!/usr/bin/env python3
import os
import asyncio
import time
import csv
import httpx
import argparse
import re
from datetime import datetime
from typing import Dict, Tuple, List
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI  # Added import for Azure OpenAI client
import uuid

# Load environment variables from .env file
load_dotenv()

# Constants
MAKEHUB_URL = os.getenv("MAKEHUB_API_URL", "https://api.makehub.ai")
MAKEHUB_API_KEY = os.getenv("MAKEHUB_API_KEY")
MODEL_ID = "openai/gpt-4o"  # MakeHub model ID
DEFAULT_TEST_PROMPT_SIZE = 10000
DEFAULT_TESTS_PER_ENDPOINT = 5
DEFAULT_OUTPUT_CSV = "makehub_raw_data.csv"
DEFAULT_MINUTES_WINDOWS = [2, 4, 8]  # Multiple time windows for MakeHub metrics

# Azure regions to test
AZURE_REGIONS = [
    "azure-eastus",
    "azure-germanywestcentral",
    "azure-norwayeast",
    "azure-swedencentral",
    "azure-switzerlandnorth",
    "azure-uksouth",
    "azure-francecentral",
]

# Mapping of MakeHub region IDs to environment variable prefixes
REGION_ENV_MAP = {
    "azure-eastus": "AOAI_STUDIO_EAST_US",
    "azure-germanywestcentral": "AOAI_STUDIO_GERMANY_WEST_CENTRAL",
    "azure-norwayeast": "AOAI_STUDIO_NORWAY_EAST",
    "azure-swedencentral": "AOAI_STUDIO_SWEDEN_CENTRAL",
    "azure-switzerlandnorth": "AOAI_STUDIO_SWITZERLAND_NORTH",
    "azure-uksouth": "AOAI_STUDIO_UK_SOUTH",
    "azure-francecentral": "AOAI_STUDIO_FRANCE_CENTRAL",
}

# OpenAI setup
MODEL_NAME = "gpt-4o"  # OpenAI model name
API_VERSION = "2024-10-21"

# Regex to detect end of sentence (period, question mark, or exclamation
# followed by space or end of string)
SENTENCE_END_PATTERN = re.compile(
    r"[.!?](\s|$)"  # Match ending punctuation + space or line end
)

# Sample French prompt for testing token caching
SAMPLE_DYNAMIC_PROMPT = """
#### Role: Tu es le réceptionniste téléphonique à l'accueil du laboratoire médical BIOANALYSES.

#### Style:
- Tu es sympathique et toujours prêt à aider, avec un sourire dans la voix.
- Tu es poli, courtois, et fais preuve d'empathie envers chaque patient.
- Tu parles français couramment et utilises un langage clair et précis,
  adapté à chaque interlocuteur.
- Tu dois être professionnel, précis et courtois dans les réponses que tu
  apportes, en respectant les protocoles du laboratoire.

#### Contexte
- La date d'aujourd'hui est {current_date}.
- L'heure actuelle est {current_time}.
- Tu es dans un appel téléphonique, cela veut dire que les instructions que
  tu recois sont issues d'une transcription, qui n'est pas toujours fiable.
  Si tu percois des erreurs, elles sont surement liées à erreurs de
  transcriptions.
- BIOANALYSES est un laboratoire d'analyse médicale reconnu pour son
  excellence et sa précision dans les résultats.
- Le laboratoire offre une gamme complète de services, y compris des
  analyses de sang, des tests génétiques, et des examens de routine.
- Nous avons récemment introduit des tests avancés pour le dépistage précoce
  de maladies chroniques.

#### Outils disponibles:
- PriseRendezVous: Cette fonction permet de prendre, modifier ou annuler un
  rendez-vous pour un patient. Paramètres: nom_patient, date_souhaitee,
  type_analyse, action(prendre/modifier/annuler).
- ConsultationResultats: Cette fonction permet de consulter les résultats
  d'analyses médicales d'un patient. Paramètres: nom_patient, id_analyse,
  date_analyse.
- EnregistrementPatient: Cette fonction permet d'enregistrer un nouveau
  patient dans le système. Paramètres: nom_patient, date_naissance, adresse.
- MiseAJourContact: Cette fonction permet de mettre à jour les informations
  de contact d'un patient. Paramètres: nom_patient, nouveau_numero,
  nouvelle_adresse.

#### Objectif: Ton objectif est de qualifier la raison de l'appel de
l'interlocuteur et de transférer à la bonne tâche.

#### Instructions à suivre:
1. Essaye de comprendre ce pour quoi l'appelant appelle. Voici les
   possibilités :
- L'appelant appelle pour fixer une date de Rendez-vous, ou de modifier ou
  annuler sa date de rendez-vous déjà fixée - va IMMÉDIATEMENT à l'étape
  DateConsultation sans rien dire !
- L'appelant appelle pour avoir des informations sur ses résultats d'analyse
  avec le laboratoire - va IMMÉDIATEMENT à l'étape InformationsConsultation
  sans rien dire !
- L'appelant appelle pour autre chose que ces deux raisons précédentes. Dis
  lui que tu ne peux aider que pour les deux raisons ci-dessus, mais que pour
  d'autres taches, tu peux rediriger vers un autre agent.

#### Appels de fonctions:
- ne mentionne jamais les fonctions que tu appelles. un message sera envoyer
  automatiquement envoyé.
- n'annonce jamais ce que tu vas faire, fais le.

#### Transitions:
- Ne demande jamais à ton interlocuteur de patienter avant une transition,
  fais la directement, en allant vers la bonne tache.

#### Attention:
- Ecris tous les chiffres et nombres en toutes lettres et ne fait jamais
  d'abbréviations pour qu'elle puissent etre synthetisés correctement.
- Respectes bien toutes les étapes ci-dessus.

#### Additional Information:
- Le laboratoire est ouvert du lundi au vendredi de 8h à 18h, et le samedi
  de 9h à 13h.
- Les résultats des analyses sont généralement disponibles sous 48 heures.
- En cas de questions supplémentaires, les patients peuvent contacter le
  service client au numéro indiqué sur leur fiche d'inscription.
- Le laboratoire utilise des technologies de pointe pour garantir la
  précision et la fiabilité des résultats.
- Tous les membres du personnel sont formés pour offrir un service client
  exceptionnel et répondre à toutes les questions des patients.
- Nous avons un programme de fidélité pour les patients réguliers, offrant
  des réductions sur certains services.
- Le laboratoire participe à des recherches cliniques et collabore avec des
  hôpitaux pour améliorer les soins de santé.
- Des ateliers éducatifs sont organisés régulièrement pour informer les
  patients sur la prévention des maladies.
"""


class AzureEndpoint:
    """Simple class to represent an Azure OpenAI endpoint"""

    def __init__(self, region_id: str):
        """Initialize with region ID"""
        self.region_id = region_id
        prefix = REGION_ENV_MAP[region_id]
        self.url = os.getenv(f"{prefix}_ENDPOINT")
        self.key = os.getenv(f"{prefix}_API_KEY")
        self.client = None

        if not self.url or not self.key:
            raise ValueError(f"Missing environment variables for {region_id}")

    async def get_client(self):
        """Get Azure OpenAI client for this endpoint"""
        if not self.client:
            self.client = AsyncAzureOpenAI(
                api_key=self.key, api_version=API_VERSION, azure_endpoint=self.url
            )
        return self.client


async def get_makehub_metrics(
    model_id: str, n_last_minutes: int, max_retries: int = 3
) -> Dict:
    """Get metrics from MakeHub API

    Args:
        model_id: Model ID in MakeHub format
        n_last_minutes: Time window to get metrics for
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary of metrics data by endpoint
    """
    if not MAKEHUB_API_KEY:
        print("Error: MAKEHUB_API_KEY environment variable not set")
        return {"error": "MAKEHUB_API_KEY not set"}

    url = (
        f"{MAKEHUB_URL}/v1/metrics?model_id={model_id}&"
        f"n_last_minutes={n_last_minutes}"
    )
    headers = {
        "Authorization": f"Bearer {MAKEHUB_API_KEY}",
        "Content-Type": "application/json",
    }

    retry_count = 0
    while retry_count <= max_retries:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                print(f"Requesting MakeHub metrics for {n_last_minutes} minutes...")
                print(f"Request URL: {url}")
                response = await client.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    print(f"MakeHub response for {n_last_minutes} min window received")
                    print(f"Response data: {data}")  # Log the response data

                    # Check if we have any actual latency values
                    has_latency = False
                    latency_field = f"avg_latency_{n_last_minutes}min_ms"

                    # Log missing metrics for endpoints
                    for region_id in AZURE_REGIONS:
                        if region_id not in data:
                            print(
                                f"  WARNING: No metrics for endpoint {region_id} "
                                f"in {n_last_minutes}min window"
                            )
                        elif not isinstance(data[region_id], dict):
                            print(
                                f"  WARNING: Metrics for {region_id} in "
                                f"{n_last_minutes}min window "
                                f"is not a dictionary: {data[region_id]}"
                            )
                        elif latency_field not in data[region_id]:
                            print(
                                f"  WARNING: No latency data for {region_id} "
                                f"in {n_last_minutes}min window"
                            )

                    for region, metrics in data.items():
                        if isinstance(metrics, dict) and latency_field in metrics:
                            has_latency = True
                            print(
                                f"  Region {region} has latency: "
                                f"{metrics.get(latency_field)}ms"
                            )

                    if not has_latency:
                        print("  No latency data found in response for any region")
                        if retry_count < max_retries:
                            retry_count += 1
                            delay = 2 + retry_count * 2  # Increasing delay
                            print(f"  Retrying in {delay} seconds...")
                            await asyncio.sleep(delay)
                            continue

                    return data
                else:
                    print(
                        f"Failed to fetch metrics for {n_last_minutes}min window: "
                        f"{response.text}"
                    )
                    print(
                        f"Response status code: {response.status_code}"
                    )  # Log the status code
                    if retry_count < max_retries:
                        retry_count += 1
                        delay = 2 + retry_count * 2  # Increasing delay
                        print(f"  Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                    return {"error": response.text}

        except Exception as e:
            print(
                f"Error fetching MakeHub metrics for {n_last_minutes}min window: {str(e)}"
            )
            if retry_count < max_retries:
                retry_count += 1
                delay = 2 + retry_count * 2  # Increasing delay
                print(f"  Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                continue
            return {"error": str(e)}

        break  # If we got here without continuing, we're done

    return {"error": f"Failed after {max_retries} retry attempts"}


async def send_test_completion(
    endpoint: AzureEndpoint, prompt_text: str
) -> Tuple[float, bool, Dict, float, float, str, str]:
    """Send a test completion to an endpoint and measure latency

    Args:
        endpoint: The endpoint to test
        prompt_text: The prompt text to send

    Returns:
        Tuple of (latency in seconds, success boolean, response data,
                  time to first token, time to first sentence,
                  request start time ISO string, request end time ISO string)
    """
    client = await endpoint.get_client()
    success = False
    response_data = {}
    ttft = None  # Time to first token
    ttfs = None  # Time to first sentence
    accumulated_text = ""

    try:
        # Create messages for the completion
        messages = [
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": "Allo, bonjour, je voudrais prendre rendez-vous.",
            },
        ]

        # Start streaming completion with caching disabled
        request_start_time = datetime.now().isoformat()
        start_time = time.perf_counter()
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=100,
            stream=True,
            user=f"makehub_test_{uuid.uuid4()}",  # Unique user ID to prevent caching
        )

        success = True
        tokens_received = 0
        total_completion_tokens = 0
        found_first_sentence = False

        # Process streaming response
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if tokens_received == 0:
                    ttft = time.perf_counter() - start_time

                tokens_received += 1
                accumulated_text += delta.content

                # Check for first complete sentence if not found yet
                if not found_first_sentence and SENTENCE_END_PATTERN.search(
                    accumulated_text
                ):
                    ttfs = time.perf_counter() - start_time
                    found_first_sentence = True

            total_completion_tokens += 1

            if chunk.choices[0].finish_reason:
                response_data["finish_reason"] = chunk.choices[0].finish_reason

        # If we never found a complete sentence, use the total time
        if not found_first_sentence and accumulated_text:
            ttfs = time.perf_counter() - start_time

        # Calculate final latency
        end_time = time.perf_counter()
        request_end_time = datetime.now().isoformat()
        latency = end_time - start_time

        # Populate response data
        response_data = {
            "completion_tokens": total_completion_tokens,
            "prompt_tokens": 1094,  # Not directly available from streaming chunks
            "total_tokens": 1094 + total_completion_tokens,  # Will be calculated by adding prompt and completion
            "tokens_received": tokens_received,
            "time_to_first_token": ttft,
            "time_to_first_sentence": ttfs,
            "total_latency": latency,
        }

        # Log the endpoint data processing
        print(f"Processing endpoint data for region: {endpoint.region_id}")

        # Log the completion results
        print(
            f"Completion results for {endpoint.region_id}: Latency: {latency}, Success: {success}"
        )
        print(f"Response data: {response_data}")

        return (
            latency,
            success,
            response_data,
            ttft,
            ttfs,
            request_start_time,
            request_end_time,
        )

    except Exception as e:
        request_end_time = datetime.now().isoformat()
        print(f"Exception on {endpoint.region_id}: {str(e)}")
        response_data = {"error": str(e)}
        return (
            time.perf_counter() - start_time,
            success,
            response_data,
            ttft,
            ttfs,
            request_start_time,
            request_end_time,
        )


def generate_dummy_prompt():
    """Generate a dynamic prompt with current date and time to avoid caching

    Returns:
        A prompt string with dynamic date/time content
    """
    now = datetime.now()
    current_date = now.strftime("%d %B %Y")
    current_time = now.strftime("%H:%M:%S")

    return SAMPLE_DYNAMIC_PROMPT.format(
        current_date=current_date, current_time=current_time
    )


def generate_prompt(size: int = None, use_dummy: bool = None) -> str:
    """Generate a prompt with the current date and time"""
    return generate_dummy_prompt()


def initialize_csv(
    csv_path: str, continue_existing: bool = True, windows: List[int] = None
) -> bool:
    """Initialize the CSV file for output

    Args:
        csv_path: Path to the CSV file
        continue_existing: Whether to continue with existing file
        windows: List of window sizes to use for column names

    Returns:
        True if new file was created, False if continuing existing file
    """
    # Use default windows if none provided
    if windows is None:
        windows = DEFAULT_MINUTES_WINDOWS

    file_exists = os.path.isfile(csv_path)

    # Check if we should continue with existing file
    if file_exists and continue_existing:
        return False

    # Create new file with headers - these will be dynamically generated based on windows
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Create base columns (not window-specific)
        base_columns = [
            "timestamp",
            "request_start_time",
            "request_end_time",
            "region_id",
            "test_num",
            "prompt_size",
        ]

        # Create window-specific columns for each window size
        window_columns = []
        for window in windows:
            window_columns.extend(
                [
                    f"makehub_{window}min_window_size",
                    f"makehub_{window}min_avg_latency_ms",
                    f"makehub_{window}min_last_latency_ms",
                    f"makehub_{window}min_latency_variance_ms",
                    f"makehub_{window}min_avg_throughput_tokens_per_sec",
                    f"makehub_{window}min_last_throughput_tokens_per_sec",
                    f"makehub_{window}min_throughput_variance",
                    f"makehub_{window}min_rtt_ms",
                    f"makehub_{window}min_dt_since_last_ms",
                ]
            )

        # Create result columns
        result_columns = [
            "actual_latency_sec",
            "time_to_first_token_sec",
            "time_to_first_sentence_sec",
            "success",
            "tokens_received",
            "completion_tokens",
            "prompt_tokens",
            "total_tokens",
            "finish_reason",
            "error",
        ]

        # Write all columns
        writer.writerow(base_columns + window_columns + result_columns)
    return True


def write_test_result_to_csv(
    csv_path: str,
    timestamp: str,
    request_start_time: str,
    request_end_time: str,
    region_id: str,
    test_num: int,
    prompt_size: int,
    makehub_metrics_by_window: Dict[int, Dict],
    actual_latency: float,
    ttft: float,
    ttfs: float,
    success: bool,
    response_data: Dict,
):
    """Write a single test result to the CSV file

    Args:
        csv_path: Path to the CSV file
        timestamp: Timestamp of the test
        request_start_time: ISO format timestamp when request was initiated
        request_end_time: ISO format timestamp when request completed
        region_id: Region ID
        test_num: Test number
        prompt_size: Size of the prompt
        makehub_metrics_by_window: Dict mapping window time to metrics for this region
        actual_latency: Actual latency measured (seconds)
        ttft: Time to first token (seconds)
        ttfs: Time to first sentence (seconds)
        success: Whether the test was successful
        response_data: Additional response data
    """
    # Prepare base row data
    row_data = [
        timestamp,
        request_start_time,
        request_end_time,
        region_id,
        test_num,
        prompt_size,
    ]

    # Add data for each window in makehub_metrics_by_window
    # Only write data for windows that we actually have data for
    for window in DEFAULT_MINUTES_WINDOWS:
        metrics = makehub_metrics_by_window.get(window, {})

        # Skip writing data for this window if metrics is empty
        if not metrics:
            # Add empty values for this window to maintain CSV structure
            row_data.extend(["", "", "", "", "", "", "", "", ""])
            continue

        # Field names based on window
        avg_latency_field = f"avg_latency_{window}min_ms"
        last_latency_field = "last_latency_ms"
        latency_variance_field = f"latency_variance_{window}min_ms"
        avg_throughput_field = f"avg_throughput_{window}min_tokens_per_second"
        last_throughput_field = "last_throughput_tokens_per_second"
        throughput_variance_field = f"throughput_variance_{window}min_tokens_per_second"
        rtt_field = "rtt_from_makehub_ms"
        dt_field = "dt_since_last_measurement_ms"

        # Extract values with defaults
        row_data.extend(
            [
                window,
                metrics.get(avg_latency_field, ""),
                metrics.get(last_latency_field, ""),
                metrics.get(latency_variance_field, ""),
                metrics.get(avg_throughput_field, ""),
                metrics.get(last_throughput_field, ""),
                metrics.get(throughput_variance_field, ""),
                metrics.get(rtt_field, ""),
                metrics.get(dt_field, ""),
            ]
        )

    # Add test results
    row_data.extend(
        [
            actual_latency,
            ttft if ttft is not None else "",
            ttfs if ttfs is not None else "",
            "TRUE" if success else "FALSE",
            response_data.get("tokens_received", ""),
            response_data.get("completion_tokens", ""),
            response_data.get("prompt_tokens", ""),
            response_data.get("total_tokens", ""),
            response_data.get("finish_reason", ""),
            response_data.get("error", ""),
        ]
    )

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

    # Log the CSV writing process
    print(f"Writing results to CSV for region: {region_id}, Test number: {test_num}")


async def fetch_all_makehub_metrics(windows: List[int]) -> Dict[int, Dict]:
    """Fetch MakeHub metrics for all window sizes with delays between requests

    Args:
        windows: List of window sizes in minutes

    Returns:
        Dictionary mapping window size to metrics data
    """
    print(f"Fetching MakeHub metrics for all window sizes: {windows}...")

    # Create tasks for all window sizes with delays to avoid rate limiting
    metrics_by_window = {}
    missing_metrics = {}  # Track missing metrics by region and window

    for window in windows:
        # Add delay between requests to avoid rate limiting
        if window != windows[0]:  # No delay for first request
            delay = 1.0  # 1 second delay between window requests
            print(f"  Waiting {delay} seconds before next window request...")
            await asyncio.sleep(delay)

        result = await get_makehub_metrics(MODEL_ID, window)
        metrics_by_window[window] = result

        # Check for errors
        if "error" in result:
            print(
                f"  Error fetching MakeHub data for {window}min window: {result['error']}"
            )
            missing_metrics[window] = set(AZURE_REGIONS)  # All regions are missing
            continue

        # Track missing metrics for each region
        missing_regions = set()
        latency_field = f"avg_latency_{window}min_ms"

        for region_id in AZURE_REGIONS:
            if (
                region_id not in result
                or not isinstance(result[region_id], dict)
                or latency_field not in result[region_id]
            ):
                missing_regions.add(region_id)

        missing_metrics[window] = missing_regions

        # Check if we have any actual latency values
        has_data = False
        for region, metrics in result.items():
            if isinstance(metrics, dict) and latency_field in metrics:
                has_data = True
                break

        if has_data:
            print(f"  Successfully retrieved MakeHub data for {window}min window")
            if missing_regions:
                print(
                    f"  Missing metrics for {len(missing_regions)}/{len(AZURE_REGIONS)} regions in {window}min window:"
                )
                for region in missing_regions:
                    print(f"    - {region}")
        else:
            print(f"  No usable data in MakeHub response for {window}min window")

    # Print summary
    successful_windows = [
        w for w, data in metrics_by_window.items() if "error" not in data
    ]
    print(
        f"Successfully fetched data for {len(successful_windows)}/{len(windows)} windows"
    )

    # Print a summary of missing metrics across all windows
    print("\nSummary of missing metrics by region:")
    region_missing_count = {region: 0 for region in AZURE_REGIONS}
    for window, missing_set in missing_metrics.items():
        for region in missing_set:
            region_missing_count[region] += 1

    for region, count in region_missing_count.items():
        if count > 0:
            print(f"  {region}: Missing metrics in {count}/{len(windows)} windows")

    return metrics_by_window


def generate_output_filename(args):
    """Generate a unique output filename based on parameters and current date/time"""
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    # Include key parameters in filename
    window_str = "".join([str(w) for w in args.windows])

    # Create filename with format:
    # makehub_data_<date>_<time>_win<windows>_n<tests>.csv
    filename = (
        f"makehub_data_{date_str}_{time_str}_" f"win{window_str}_n{args.tests}.csv"
    )

    return filename


async def main(args):
    # Generate unique output CSV name
    if args.output == DEFAULT_OUTPUT_CSV:  # Only generate if using default
        csv_path = generate_output_filename(args)
    else:
        csv_path = args.output

    print(f"Output will be saved to: {csv_path}")

    # Initialize CSV - always create new file when using generated filename
    continue_existing = False  # Always create a new file
    new_file = initialize_csv(csv_path, continue_existing, args.windows)

    if new_file:
        print(f"Created new output file: {csv_path}")
    else:
        print(f"Continuing with existing output file: {csv_path}")

    print("MakeHub Raw Data Collection")
    print("==========================")
    print(f"Testing {len(AZURE_REGIONS)} Azure regions with {args.tests} tests each")
    print(f"MakeHub metrics windows: {args.windows} minutes")
    print(f"Global time between iterations: {args.global_time} seconds")
    print(f"Output will be saved to: {csv_path}")
    print()

    # Initialize endpoints
    endpoints = []
    for region_id in AZURE_REGIONS:
        try:
            endpoint = AzureEndpoint(region_id)
            endpoints.append(endpoint)
        except ValueError as e:
            print(f"Skipping {region_id}: {str(e)}")

    if not endpoints:
        print("No valid endpoints found. Please check your environment variables.")
        exit(1)

    print(f"Successfully initialized {len(endpoints)} endpoints\n")

    # Run tests
    for test_num in range(1, args.tests + 1):
        # Record the start time of this iteration
        iteration_start_time = time.perf_counter()

        print(f"Test iteration {test_num}/{args.tests}:")

        # 1. Fetch MakeHub metrics for all window sizes simultaneously
        metrics_by_window = await fetch_all_makehub_metrics(args.windows)

        # Check if we have valid data for at least one window
        valid_windows = [
            window for window, data in metrics_by_window.items() if "error" not in data
        ]

        if not valid_windows:
            print("No valid MakeHub data for any window size, skipping iteration")
            continue

        print(f"Retrieved MakeHub data for {len(valid_windows)} window sizes")

        # 2. Prepare prompts - Generate a fresh prompt for each endpoint with current time
        print("Preparing prompts with current date/time...", end="", flush=True)
        test_prompts = {}
        for endpoint in endpoints:
            # Always use generate_dummy_prompt() which refreshes dynamic variables
            test_prompts[endpoint.region_id] = generate_dummy_prompt()
        print(" Done")

        # 3. Run Azure OpenAI calls for all endpoints simultaneously
        print("Executing LLM calls to all regions...")

        # Prepare all tasks
        call_tasks = []
        for endpoint in endpoints:
            task = send_test_completion(endpoint, test_prompts[endpoint.region_id])
            call_tasks.append(task)

        # Use asyncio.gather to truly run them in parallel
        print("Starting parallel execution of all API calls...")
        results = await asyncio.gather(*call_tasks)

        # Process results
        completion_results = {}
        for i, endpoint in enumerate(endpoints):
            (
                latency,
                success,
                response_data,
                ttft,
                ttfs,
                req_start_time,
                req_end_time,
            ) = results[i]
            completion_results[endpoint.region_id] = {
                "latency": latency,
                "success": success,
                "response_data": response_data,
                "ttft": ttft,
                "ttfs": ttfs,
                "request_start_time": req_start_time,
                "request_end_time": req_end_time,
            }
            print(".", end="", flush=True)

        # 4. Write results to CSV
        print("\nWriting results to CSV...")
        timestamp = datetime.now().isoformat()

        # First, check for endpoints with no metrics across all windows
        missing_endpoint_data = {}
        for endpoint_id in [e.region_id for e in endpoints]:
            missing_windows = []
            for window in args.windows:
                if (
                    window not in metrics_by_window
                    or "error" in metrics_by_window[window]
                    or endpoint_id not in metrics_by_window[window]
                    or not isinstance(metrics_by_window[window][endpoint_id], dict)
                    or f"avg_latency_{window}min_ms"
                    not in metrics_by_window[window][endpoint_id]
                ):
                    missing_windows.append(window)

            if missing_windows:
                missing_endpoint_data[endpoint_id] = missing_windows
                print(
                    f"WARNING: Endpoint {endpoint_id} missing metrics for {len(missing_windows)}/{len(args.windows)} windows:"
                )
                for window in missing_windows:
                    print(f"  - {window}min window")

        if not missing_endpoint_data:
            print("All endpoints have metrics for all windows!")

        for endpoint_id, result in completion_results.items():
            # Get MakeHub metrics for each window for this region
            endpoint_metrics_by_window = {}
            for window, metrics in metrics_by_window.items():
                # Only include metrics if they exist and are valid for this endpoint
                if "error" not in metrics and endpoint_id in metrics:
                    if isinstance(metrics[endpoint_id], dict):
                        # Check if we have actual latency values in this window for this endpoint
                        latency_field = f"avg_latency_{window}min_ms"
                        if latency_field in metrics[endpoint_id]:
                            endpoint_metrics_by_window[window] = metrics[endpoint_id]
                            print(
                                f"  Found valid metrics for {endpoint_id} in "
                                f"{window}min window"
                            )
                        else:
                            print(
                                f"  No latency field for {endpoint_id} in "
                                f"{window}min window"
                            )
                    else:
                        print(
                            f"  Metrics for {endpoint_id} in {window}min window is "
                            f"not a dict: {metrics[endpoint_id]}"
                        )
                else:
                    if "error" in metrics:
                        print(
                            f"  Error in metrics for {window}min window: "
                            f"{metrics['error']}"
                        )
                    else:
                        print(f"  No data for {endpoint_id} in {window}min window")

            # Use a fixed prompt_tokens value (approximately 1000 tokens) for the dynamic prompt
            prompt_tokens = 1000

            # Write the result to CSV
            write_test_result_to_csv(
                csv_path,
                timestamp,
                result["request_start_time"],
                result["request_end_time"],
                endpoint_id,
                test_num,
                prompt_tokens,  # Use the token count instead of character count
                endpoint_metrics_by_window,
                result["latency"],
                result["ttft"],
                result["ttfs"],
                result["success"],
                result["response_data"],
            )

        print(f"Completed iteration {test_num}/{args.tests}")

        # Calculate how much time to wait to maintain consistent global time between iterations
        if test_num < args.tests:
            iteration_elapsed = time.perf_counter() - iteration_start_time
            wait_time = max(0, args.global_time - iteration_elapsed)
            if wait_time > 0:
                print(
                    f"Waiting {wait_time:.2f} seconds to maintain {args.global_time} second spacing between iterations...\n"
                )
                await asyncio.sleep(wait_time)
            else:
                print(
                    f"Iteration took longer than global time setting ({iteration_elapsed:.2f}s > {args.global_time}s), continuing immediately\n"
                )

    print("\nData collection complete!")
    print(f"Raw data saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect raw data comparing MakeHub metrics with actual Azure "
        "OpenAI latencies"
    )
    parser.add_argument(
        "--tests",
        type=int,
        default=DEFAULT_TESTS_PER_ENDPOINT,
        help=f"Number of test iterations to run (default: {DEFAULT_TESTS_PER_ENDPOINT})",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_MINUTES_WINDOWS,
        help=f"MakeHub metrics windows in minutes (default: {DEFAULT_MINUTES_WINDOWS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV file path (default: auto-generated)",
    )
    parser.add_argument(
        "--global-time",
        type=float,
        default=60.0,
        help="Global time between iteration starts in seconds (default: 60.0)",
    )
    args = parser.parse_args()

    # Set default prompt_size for backward compatibility with other code
    args.prompt_size = 1000

    asyncio.run(main(args))
