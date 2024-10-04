import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# Set up the random number generator
rng = np.random.Generator(np.random.PCG64(42))  # 42 is the seed

# Set up Faker
fake = Faker()
Faker.seed(42)  # Set seed for reproducibility

# Generate accounts data
num_companies = 250

def generate_accounts_data() -> pd.DataFrame:
    accounts_data = []
    for i in range(1, num_companies + 1):
        company_name = fake.company()
        company_id = f"COMP{i:03d}"
        first_name = fake.first_name()
        last_name = fake.last_name()
        account_owner = f"{first_name} {last_name}"
        email = f"{first_name.lower()}.{last_name.lower()}@{company_name.lower().replace(' ', '')}.com" # noqa: E501
        seats_available = rng.integers(2, 21)
        contract_start_date = datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 366)))
        contract_size = int(rng.integers(1, 11)) * 1000
        churned = rng.random() < 0.2
        start_date = contract_start_date - timedelta(days=int(rng.integers(0, 31)))
        renewal_or_churn_date = contract_start_date + timedelta(days=365)

        accounts_data.append({
            "Company ID": company_id,
            "Company Name": company_name,
            "Account Owner": account_owner,
            "Account Owner Email": email,
            "Seats Available": int(seats_available),
            "Contract Start Date": contract_start_date,
            "Contract Size per Year": contract_size,
            "Churned": churned,
            "Start Date": start_date,
            "Did Churn": "Y" if churned else "N",
            "Renewal or Churn Date": renewal_or_churn_date,
        })
    return pd.DataFrame(accounts_data)

def generate_session_flow(
    is_churned: bool,  #noqa 
) -> list[tuple[str,str]]:
    pages = ["homepage", "report_generation", "data_tables"]
    flow = []

    # Randomly generate max rows for this session (1 to 6)
    max_rows = rng.integers(1, 7)

    current_page = "homepage"
    flow.append(("homepage", "enter"))

    while len(flow) < max_rows:
        if current_page == "homepage":
            # Higher chance to end session for churned users
            if is_churned and rng.random() < 0.4:  # 40% chance to end session for churned users
                break
            if rng.random() < 0.2:  # 20% chance to end session for non-churned users
                break

            # If not ending session, always click button and go to another page
            flow.append(("homepage", "click_button"))
            next_page = rng.choice(["report_generation", "data_tables"])

        else:  # On report_generation or data_tables pages
            # Chance to end session
            if rng.random() < 0.3:  # 30% chance to end session
                break

            # If not ending session, decide between clicking button or starting task
            if rng.random() < 0.7:  # 70% chance to click button
                flow.append((current_page, "click_button"))
                # Go to the other non-homepage page
                next_page = next(p for p in pages if p != current_page and p != "homepage")
            else:  # 30% chance to start task
                # Churned users are less likely to start tasks
                if not is_churned or (is_churned and rng.random() < 0.3):
                    flow.append((current_page, "start_task"))
                    next_page = current_page  # Stay on the same page after starting a task
                else:
                    # If churned user doesn't start task, end session
                    break

        # Add enter event for the next page if it's different from the current page
        if next_page != current_page and len(flow) < max_rows:
            flow.append((next_page, "enter"))

        current_page = next_page

        # Check if we've reached max_rows
        if len(flow) >= max_rows:
            break

    return flow

def generate_events_data(
    accounts_df: pd.DataFrame,
) -> pd.DataFrame:
    events_data = []

    for _, account in accounts_df.iterrows():
        company_id = account["Company ID"]
        total_seats = account["Seats Available"]
        is_churned = account["Churned"]

        # Determine number of active users (allowing for 0)
        if is_churned:
            max_active_users = max(0, int(rng.beta(2, 5) * total_seats))  # Right-skewed for churned
        else:
            max_active_users = max(0, int(rng.beta(5, 2) * total_seats))  # Left-skewed for non-churned # noqa: E501

        active_users = rng.integers(0, max_active_users + 1)
        user_ids = [f"{company_id}_USER{i:02d}" for i in range(1, active_users + 1)]
        print (f"number of users for this account: {len(user_ids)}")

        # Ensure all usage happens within a year of renewal_or_churn_date
        usage_start_date = max(account["Start Date"], account["Renewal or Churn Date"] - timedelta(days=365)) # noqa: E501
        usage_period = (account["Renewal or Churn Date"] - usage_start_date).days

        for user_id in user_ids:
            # Determine number of sessions for this user (up to ~365, but less for churned on average) # noqa: E501
            if is_churned:
                max_sessions = int(rng.beta(2, 5) * 90)  # Right-skewed for churned
            else:
                max_sessions = int(rng.beta(5, 2) * 90)  # Left-skewed for non-churned

            num_sessions = max(0, int(rng.normal(max_sessions / 2, max_sessions / 4)))
            print (f"number of sessions for this account: {num_sessions}")

            for _ in range(num_sessions):
                session_id = f"{company_id}_SESSION{rng.integers(1, 1001):04d}"

                # Generate session start times
                session_starts = []
                for _ in range(num_sessions):
                    if is_churned:
                        # More likely to be earlier in the period for churned accounts
                        day_offset = int(rng.beta(1, 4) * usage_period)
                    else:
                        # Slightly more sessions later in the period for non-churned accounts
                        day_offset = int(rng.beta(1.2, 1) * usage_period)

                    session_start = usage_start_date + timedelta(days=day_offset)
                    if session_start <= account["Renewal or Churn Date"]:
                        session_starts.append(session_start)

                # Sort session start times
                session_starts.sort()

                # Generate events for each session
                for session_start in session_starts:
                    session_id = f"{company_id}_SESSION{rng.integers(1, 1001):04d}"
                    session_flow = generate_session_flow(is_churned)
                    current_time = session_start

                    for page, event in session_flow:
                        # Add a realistic time gap before the event
                        if event == "enter":
                            current_time += timedelta(seconds=int(rng.integers(1, 10000)))
                        elif event == "click_button":
                            current_time += timedelta(seconds=int(rng.integers(10, 20000)))
                        elif event == "start_task":
                            current_time += timedelta(seconds=int(rng.integers(25, 30000)))

                        if current_time > account["Renewal or Churn Date"]:
                            break

                        events_data.append({
                            "Company ID": company_id,
                            "User ID": user_id,
                            "Session ID": session_id,
                            "Page Name": page,
                            "Event": event,
                            "Timestamp": current_time,
                        })

    return pd.DataFrame(events_data)


accounts_df = generate_accounts_data()
print ("accounts finished generating")

events_df = generate_events_data(accounts_df)
print ("events finished generating")

# Sort and reset index
events_df = events_df.sort_values("Timestamp").reset_index(drop=True)

# Display first few rows of each dataframe
print("Accounts Data:")
print(accounts_df.head())
print("\nEvents Data:")
print(events_df.head())

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Save to CSV in the current script's directory
accounts_df.to_csv(os.path.join(current_dir, "accounts_data.csv"), index=False)
events_df.to_csv(os.path.join(current_dir, "events_data.csv"), index=False)

print(f"\nData has been generated and saved to 'accounts_data.csv' and 'events_data.csv' in {current_dir}") # noqa: E501
