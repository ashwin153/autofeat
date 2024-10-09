import datetime

import faker
import numpy
import polars

from autofeat.convert.into_columns import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table

_RNG = numpy.random.Generator(numpy.random.PCG64())


_FAKE = faker.Faker()


def from_example(

) -> Dataset:
    """Load from randomized example data.

    :return: Example dataset.
    """
    accounts = _generate_accounts()
    sessions = _generate_sessions(accounts)

    return Dataset([
        Table(
            name="Accounts",
            data=accounts.lazy(),
            columns=into_columns(accounts),
        ),
        Table(
            name="Sessions",
            data=sessions.lazy(),
            columns=into_columns(sessions),
        ),
    ])


def _generate_accounts(
    *,
    count: int = 250,
) -> polars.DataFrame:
    accounts = []

    for i in range(1, count + 1):
        company_name = _FAKE.company()
        company_id = f"COMP{i:03d}"
        first_name = _FAKE.first_name()
        last_name = _FAKE.last_name()
        account_owner = f"{first_name} {last_name}"
        email = f"{first_name}.{last_name}@{company_name.replace(' ', '')}.com".lower()
        seats_available = int(_RNG.integers(2, 21))
        contract_start_date_offset = datetime.timedelta(days=int(_RNG.integers(0, 366)))
        contract_start_date = datetime.datetime(2023, 1, 1) + contract_start_date_offset
        contract_size = int(_RNG.integers(1, 11)) * 1000
        churned = _RNG.random() < 0.2
        start_date_offset = datetime.timedelta(days=int(_RNG.integers(0, 31)))
        start_date = contract_start_date - start_date_offset
        renewal_or_churn_date = contract_start_date + datetime.timedelta(days=365)

        accounts.append({
            "Account Owner Email": email,
            "Account Owner": account_owner,
            "Churned": churned,
            "Company ID": company_id,
            "Company Name": company_name,
            "Contract Size per Year": contract_size,
            "Contract Start Date": contract_start_date,
            "Did Churn": "Y" if churned else "N",
            "Renewal or Churn Date": renewal_or_churn_date,
            "Seats Available": seats_available,
            "Start Date": start_date,
        })

    df = polars.DataFrame(accounts)
    return df.sort("Company ID")


def _generate_sessions(
    accounts: polars.DataFrame,
) -> polars.DataFrame:
    sessions = []

    for account in accounts.rows(named=True):
        # churned accounts tend to actively use fewer of the available seats
        active_usage = _RNG.beta(2, 5) if account["Churned"] else _RNG.beta(5, 2)
        active_seats = _RNG.integers(0, int(active_usage * account["Seats Available"]) + 1)
        active_users = [f"{account['Company ID']}_USER{i:02d}" for i in range(1, active_seats + 1)]

        # all usage happens within a year of renewal_or_churn_date
        usage_start_date = max(
            account["Start Date"],
            account["Renewal or Churn Date"] - datetime.timedelta(days=365),
        )

        usage_period = (account["Renewal or Churn Date"] - usage_start_date).days

        for user_id in active_users:
            # churned users tend to have fewer sessions
            max_sessions = (_RNG.beta(2, 5) if account["Churned"] else _RNG.beta(5, 2)) * 90
            num_sessions = max(0, int(_RNG.normal(max_sessions / 2, max_sessions / 4)))

            for _ in range(num_sessions):
                session_starts = []
                for _ in range(num_sessions):
                    # churned sessions tend to have started earlier in the usage period
                    session_start_offset = (
                        int(_RNG.beta(1, 4) * usage_period)
                        if account["Churned"]
                        else int(_RNG.beta(1.2, 1) * usage_period)
                    )

                    session_start = usage_start_date + datetime.timedelta(days=session_start_offset)
                    session_starts.append(session_start)

                for session_start in sorted(session_starts):
                    session_id = f"{account['Company ID']}_SESSION{_RNG.integers(1, 1001):04d}"
                    session = _generate_session(is_churned=account["Churned"])

                    event_timestamp = session_start
                    for page, event, event_offset in session:
                        event_timestamp += datetime.timedelta(seconds=event_offset)
                        if event_timestamp > account["Renewal or Churn Date"]:
                            break

                        sessions.append({
                            "Company ID": account["Company ID"],
                            "Event": event,
                            "Page Name": page,
                            "Session ID": session_id,
                            "Timestamp": event_timestamp,
                            "User ID": user_id,
                        })

    df = polars.DataFrame(sessions)
    return df.sort("Timestamp")


def _generate_session(
    *,
    is_churned: bool,
) -> list[tuple[str, str, int]]:
    session = []

    # begin session on the home page
    pages = ["home_page", "report_generation", "data_tables"]
    page = pages[0]
    session.append((page, "enter", int(_RNG.integers(1, 10000))))

    # 40% chance to immediately end session if churned, 20% otherwise
    if _RNG.random() < (0.4 if is_churned else 0.2):
        return session

    while len(session) < _RNG.integers(1, 7):
        if _RNG.random() < 0.7:
            # 70% chance of navigating to another page
            session.append((page, "click_button", int(_RNG.integers(10, 20000))))
            page = _RNG.choice([p for p in pages if p != page])
            session.append((page, "enter", int(_RNG.integers(1, 10000))))
        elif _RNG.random() < (0.3 if is_churned else 0.9):
            # 30% chance to start a task if churned, 90% otherwise
            session.append((page, "start_task", int(_RNG.integers(25, 30000))))
        else:
            # end session
            break

    return session
