"""
Appointment Service — database operations for the agent swarm.
Agents call these functions to read/write the appointments table.
"""
from zoneinfo import ZoneInfo
CLINIC_TZ = ZoneInfo("America/Toronto")
from datetime import datetime, timedelta
from api.services.supabase_client import get_supabase_admin, log_audit_event
from api.core.config import settings
import json
import logging

logger = logging.getLogger(__name__)

# Default appointment durations by type (minutes)
APPOINTMENT_DURATIONS = {
    "cleaning": 60,
    "exam": 30,
    "filling": 45,
    "crown": 90,
    "root_canal": 90,
    "extraction": 60,
    "whitening": 60,
    "consultation": 30,
    "emergency": 30,
    "follow_up": 15,
    "general": 30,
}

# Business hours
BUSINESS_HOURS = {
    "start": 8,  # 8 AM
    "end": 17,   # 5 PM
    "slot_minutes": 30,
    "days": [0, 1, 2, 3, 4],  # Mon-Fri
}


async def lookup_patient_by_name(
    workspace_id: str,
    patient_name: str,
) -> dict:
    """
    Look up a patient by name in the encrypted patients table.
    Uses the list_patients RPC to decrypt PHI, then filters by name.
    Includes fuzzy matching to handle voice transcription errors.

    Returns:
        {
            "found": True/False,
            "patient": { "id", "external_ref", "full_name", ... } or None,
            "candidates": [ ... ] if multiple partial matches,
            "message": "human-readable status"
        }
    """
    from difflib import SequenceMatcher

    supabase = get_supabase_admin()

    try:
        result = supabase.rpc("list_patients", {
            "p_workspace_id": workspace_id,
            "p_encryption_key": settings.phi_encryption_key,
        }).execute()
    except Exception as e:
        logger.error(f"Patient lookup RPC failed: {e}")
        return {
            "found": False,
            "patient": None,
            "candidates": [],
            "message": f"Failed to query patients: {e}",
        }

    patients = result.data or []

    if not patients:
        return {
            "found": False,
            "patient": None,
            "candidates": [],
            "message": "No patients found in this workspace.",
        }

    # DEBUG: Log what the RPC returns so we can see the field names
    print(f"[PATIENT LOOKUP] Searching for: '{patient_name}'")
    print(f"[PATIENT LOOKUP] Total patients in workspace: {len(patients)}")
    if patients:
        print(f"[PATIENT LOOKUP] Sample patient keys: {list(patients[0].keys())}")
        print(f"[PATIENT LOOKUP] Sample patient data: {patients[0]}")

    # Normalize search name
    search = patient_name.strip().lower()

    # 1) Exact match (case-insensitive)
    exact = [p for p in patients if (p.get("full_name") or "").strip().lower() == search]
    if len(exact) == 1:
        print(f"[PATIENT LOOKUP] Exact match: {exact[0]['full_name']}")
        return {
            "found": True,
            "patient": _safe_patient(exact[0]),
            "candidates": [],
            "message": f"Patient found: {exact[0]['full_name']}",
        }

    # 2) Partial / contains match
    partial = [p for p in patients if search in (p.get("full_name") or "").strip().lower()]
    if len(partial) == 1:
        print(f"[PATIENT LOOKUP] Partial match: {partial[0]['full_name']}")
        return {
            "found": True,
            "patient": _safe_patient(partial[0]),
            "candidates": [],
            "message": f"Patient found: {partial[0]['full_name']}",
        }

    if len(partial) > 1:
        return {
            "found": False,
            "patient": None,
            "candidates": [_safe_patient(p) for p in partial[:5]],
            "message": f"Multiple patients match '{patient_name}'. Please clarify which one.",
        }

    # 3) First name match — useful when caller gives just first name
    search_parts = search.split()
    if search_parts:
        first_name = search_parts[0]
        first_matches = [
            p for p in patients
            if (p.get("full_name") or "").strip().lower().split()[0] == first_name
        ]
        if len(first_matches) == 1:
            print(f"[PATIENT LOOKUP] First name match: {first_matches[0]['full_name']}")
            return {
                "found": True,
                "patient": _safe_patient(first_matches[0]),
                "candidates": [],
                "message": f"Patient found: {first_matches[0]['full_name']}",
            }
        if len(first_matches) > 1:
            return {
                "found": False,
                "patient": None,
                "candidates": [_safe_patient(p) for p in first_matches[:5]],
                "message": f"Multiple patients with first name '{first_name}'. Please provide the full name.",
            }

    # 4) Fuzzy match — handles voice transcription errors (e.g. "Vialta" vs "Villalta")
    fuzzy_matches = []
    for p in patients:
        name = (p.get("full_name") or "").strip().lower()
        ratio = SequenceMatcher(None, search, name).ratio()
        if ratio >= 0.65:  # 65% similarity threshold
            fuzzy_matches.append((p, ratio))
            print(f"[PATIENT LOOKUP] Fuzzy match: '{name}' score={ratio:.2f}")

    # Sort by best match
    fuzzy_matches.sort(key=lambda x: x[1], reverse=True)

    if len(fuzzy_matches) == 1:
        best = fuzzy_matches[0][0]
        print(f"[PATIENT LOOKUP] Single fuzzy match: {best['full_name']} (score={fuzzy_matches[0][1]:.2f})")
        return {
            "found": True,
            "patient": _safe_patient(best),
            "candidates": [],
            "message": f"Patient found: {best['full_name']}",
        }

    if len(fuzzy_matches) > 1:
        # If top match is significantly better than second, use it
        if fuzzy_matches[0][1] - fuzzy_matches[1][1] >= 0.15:
            best = fuzzy_matches[0][0]
            print(f"[PATIENT LOOKUP] Best fuzzy match: {best['full_name']} (score={fuzzy_matches[0][1]:.2f}, gap={fuzzy_matches[0][1] - fuzzy_matches[1][1]:.2f})")
            return {
                "found": True,
                "patient": _safe_patient(best),
                "candidates": [],
                "message": f"Patient found: {best['full_name']}",
            }
        return {
            "found": False,
            "patient": None,
            "candidates": [_safe_patient(p) for p, _ in fuzzy_matches[:5]],
            "message": f"Multiple patients match '{patient_name}'. Please clarify which one.",
        }

    # 5) No match at all
    return {
        "found": False,
        "patient": None,
        "candidates": [],
        "message": f"No patient named '{patient_name}' found. They may need to be registered first.",
    }


def _safe_patient(patient: dict) -> dict:
    """Return only the fields safe for agent context (no raw PII beyond name)."""
    return {
        "id": patient.get("id"),
        "external_ref": patient.get("external_ref"),
        "full_name": patient.get("full_name"),
        "is_active": patient.get("is_active", True),
    }


async def get_appointments_for_date(
    workspace_id: str,
    date: str,  # "2026-02-24"
) -> list[dict]:
    """Get all appointments for a specific date."""
    supabase = get_supabase_admin()

    start = f"{date}T00:00:00Z"
    end = f"{date}T23:59:59Z"

    result = (
        supabase.table("appointments")
        .select("id, title, appointment_type, start_time, end_time, duration_minutes, status, patient_id")
        .eq("workspace_id", workspace_id)
        .gte("start_time", start)
        .lte("start_time", end)
        .neq("status", "cancelled")
        .order("start_time")
        .execute()
    )

    return result.data or []


async def get_appointments_for_range(
    workspace_id: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Get appointments for a date range."""
    supabase = get_supabase_admin()

    result = (
        supabase.table("appointments")
        .select("id, title, appointment_type, start_time, end_time, duration_minutes, status, patient_id")
        .eq("workspace_id", workspace_id)
        .gte("start_time", f"{start_date}T00:00:00Z")
        .lte("start_time", f"{end_date}T23:59:59Z")
        .neq("status", "cancelled")
        .order("start_time")
        .execute()
    )

    return result.data or []


async def get_patient_appointments(
    workspace_id: str,
    patient_id: str,
    upcoming_only: bool = True,
) -> list[dict]:
    """Get appointments for a specific patient."""
    supabase = get_supabase_admin()

    query = (
        supabase.table("appointments")
        .select("id, title, appointment_type, start_time, end_time, duration_minutes, status")
        .eq("workspace_id", workspace_id)
        .eq("patient_id", patient_id)
        .neq("status", "cancelled")
        .order("start_time")
    )

    if upcoming_only:
        query = query.gte("start_time", datetime.now(CLINIC_TZ).isoformat())

    result = query.execute()
    return result.data or []


async def check_availability(
    workspace_id: str,
    date: str,
    duration_minutes: int = 30,
) -> list[dict]:
    """
    Find available time slots for a given date and duration.
    Returns list of available slots: [{"start": "09:00", "end": "09:30"}, ...]
    """
    existing = await get_appointments_for_date(workspace_id, date)

    # Parse date to check if it's a business day
    check_date = datetime.strptime(date, "%Y-%m-%d")
    if check_date.weekday() not in BUSINESS_HOURS["days"]:
        return []  # Not a business day

    # Build list of busy periods
    busy = []
    for appt in existing:
        start = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(appt["end_time"].replace("Z", "+00:00"))
        busy.append((start.hour * 60 + start.minute, end.hour * 60 + end.minute))

    # Find free slots
    available = []
    slot_start = BUSINESS_HOURS["start"] * 60  # 8:00 = 480

    while slot_start + duration_minutes <= BUSINESS_HOURS["end"] * 60:
        slot_end = slot_start + duration_minutes

        # Check if slot overlaps with any busy period
        is_free = True
        for busy_start, busy_end in busy:
            if slot_start < busy_end and slot_end > busy_start:
                is_free = False
                break

        if is_free:
            start_h, start_m = divmod(slot_start, 60)
            end_h, end_m = divmod(slot_end, 60)
            available.append({
                "start": f"{start_h:02d}:{start_m:02d}",
                "end": f"{end_h:02d}:{end_m:02d}",
            })

        slot_start += BUSINESS_HOURS["slot_minutes"]

    return available


async def find_next_available(
    workspace_id: str,
    duration_minutes: int = 30,
    days_ahead: int = 14,
    max_results: int = 5,
) -> list[dict]:
    """
    Find the next available slots across multiple days.
    Returns: [{"date": "2026-02-25", "slots": [{"start": "09:00", "end": "09:30"}, ...]}]
    """
    results = []
    today = datetime.now(CLINIC_TZ).date()

    for day_offset in range(days_ahead):
        check_date = today + timedelta(days=day_offset)

        # Skip weekends
        if check_date.weekday() not in BUSINESS_HOURS["days"]:
            continue

        # Skip today if past business hours
        if day_offset == 0:
            now = datetime.now(CLINIC_TZ)
            if now.hour >= BUSINESS_HOURS["end"]:
                continue

        date_str = check_date.isoformat()
        slots = await check_availability(workspace_id, date_str, duration_minutes)

        # Filter out past slots for today
        if day_offset == 0:
            now = datetime.utcnow()
            current_minutes = now.hour * 60 + now.minute
            slots = [s for s in slots if int(s["start"].split(":")[0]) * 60 + int(s["start"].split(":")[1]) > current_minutes]

        if slots:
            results.append({
                "date": date_str,
                "day_name": check_date.strftime("%A"),
                "slots": slots[:3],  # Top 3 per day
            })

        if len(results) >= max_results:
            break

    return results


async def book_appointment(
    workspace_id: str,
    date: str,
    time: str,  # "09:00"
    appointment_type: str = "general",
    patient_id: str | None = None,
    patient_name: str | None = None,
    title: str | None = None,
    notes: str | None = None,
    source: str = "concierge",
) -> dict:
    """Book a new appointment."""
    supabase = get_supabase_admin()

    duration = APPOINTMENT_DURATIONS.get(appointment_type, 30)
    start_time = datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), int(time[:2]), int(time[3:5]), tzinfo=CLINIC_TZ)
    end_time = start_time + timedelta(minutes=duration)

    # Build title: "Cleaning — Julio Villalta" or just "Cleaning"
    if not title:
        type_label = appointment_type.replace("_", " ").title()
        title = f"{type_label} — {patient_name}" if patient_name else type_label

    # Verify slot is available
    slots = await check_availability(workspace_id, date, duration)
    is_available = any(s["start"] == time for s in slots)

    if not is_available:
        return {
            "success": False,
            "error": "This time slot is not available",
            "available_slots": slots[:5],
        }

    # Validate patient exists if patient_id is provided
    if patient_id:
        supabase_check = get_supabase_admin()
        patient_result = (
            supabase_check.table("patients")
            .select("id, external_ref, is_active")
            .eq("id", patient_id)
            .eq("workspace_id", workspace_id)
            .execute()
        )
        if not patient_result.data:
            return {
                "success": False,
                "error": f"Patient ID '{patient_id}' not found in this workspace. "
                         "Please verify the patient name or register them first.",
            }
        patient_record = patient_result.data[0]
        if not patient_record.get("is_active", True):
            return {
                "success": False,
                "error": "This patient record is inactive. Please reactivate before booking.",
            }

    result = (
        supabase.table("appointments")
        .insert({
            "workspace_id": workspace_id,
            "patient_id": patient_id,
            "title": title,
            "appointment_type": appointment_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration,
            "notes": notes,
            "source": source,
            "status": "scheduled",
        })
        .execute()
    )

    if result.data:
        appt = result.data[0] if isinstance(result.data, list) else result.data

        await log_audit_event(
            workspace_id=workspace_id,
            actor_type="agent",
            actor_id="concierge",
            action="appointment_booked",
            resource_type="appointment",
            resource_id=appt["id"],
            metadata={
                "date": date,
                "time": time,
                "type": appointment_type,
                "patient_id": patient_id,
                "source": source,
            },
        )

        return {
            "success": True,
            "appointment": {
                "id": appt["id"],
                "date": date,
                "time": time,
                "type": appointment_type,
                "duration": duration,
                "status": "scheduled",
            },
        }

    return {"success": False, "error": "Failed to create appointment"}


async def cancel_appointment(
    workspace_id: str,
    appointment_id: str | None = None,
    patient_id: str | None = None,
    reason: str = "Patient requested cancellation",
) -> dict:
    """
    Cancel an appointment by ID or by patient's next upcoming appointment.
    """
    supabase = get_supabase_admin()

    # Find the appointment
    if appointment_id:
        result = (
            supabase.table("appointments")
            .select("*")
            .eq("id", appointment_id)
            .eq("workspace_id", workspace_id)
            .single()
            .execute()
        )
    elif patient_id:
        # Cancel next upcoming appointment for this patient
        result = (
            supabase.table("appointments")
            .select("*")
            .eq("workspace_id", workspace_id)
            .eq("patient_id", patient_id)
            .gte("start_time", datetime.now(CLINIC_TZ).isoformat())
            .neq("status", "cancelled")
            .order("start_time")
            .limit(1)
            .single()
            .execute()
        )
    else:
        return {"success": False, "error": "Need appointment_id or patient_id"}

    if not result.data:
        return {"success": False, "error": "Appointment not found"}

    appt = result.data

    # Cancel it
    supabase.table("appointments").update({
        "status": "cancelled",
        "cancellation_reason": reason,
    }).eq("id", appt["id"]).execute()

    await log_audit_event(
        workspace_id=workspace_id,
        actor_type="agent",
        actor_id="concierge",
        action="appointment_cancelled",
        resource_type="appointment",
        resource_id=appt["id"],
        metadata={
            "reason": reason,
            "original_date": appt["start_time"],
            "type": appt["appointment_type"],
        },
    )

    # Find next available slots to offer
    duration = appt.get("duration_minutes", 30)
    next_slots = await find_next_available(workspace_id, duration, days_ahead=14, max_results=3)

    return {
        "success": True,
        "cancelled_appointment": {
            "id": appt["id"],
            "date": appt["start_time"][:10],
            "time": appt["start_time"][11:16],
            "type": appt["appointment_type"],
        },
        "suggested_reschedule": next_slots,
    }


async def reschedule_appointment(
    workspace_id: str,
    appointment_id: str,
    new_date: str,
    new_time: str,
) -> dict:
    """Reschedule an existing appointment to a new date/time."""
    supabase = get_supabase_admin()

    # Get current appointment
    current = (
        supabase.table("appointments")
        .select("*")
        .eq("id", appointment_id)
        .eq("workspace_id", workspace_id)
        .single()
        .execute()
    )

    if not current.data:
        return {"success": False, "error": "Appointment not found"}

    appt = current.data
    duration = appt.get("duration_minutes", 30)

    # Check if new slot is available
    slots = await check_availability(workspace_id, new_date, duration)
    is_available = any(s["start"] == new_time for s in slots)

    if not is_available:
        return {
            "success": False,
            "error": "The requested time is not available",
            "available_slots": slots[:5],
        }

    # Update
    new_start = datetime.fromisoformat(f"{new_date}T{new_time}:00")
    new_end = new_start + timedelta(minutes=duration)

    supabase.table("appointments").update({
        "start_time": new_start.isoformat(),
        "end_time": new_end.isoformat(),
        "status": "scheduled",
    }).eq("id", appointment_id).execute()

    await log_audit_event(
        workspace_id=workspace_id,
        actor_type="agent",
        actor_id="concierge",
        action="appointment_rescheduled",
        resource_type="appointment",
        resource_id=appointment_id,
        metadata={
            "old_date": appt["start_time"][:10],
            "old_time": appt["start_time"][11:16],
            "new_date": new_date,
            "new_time": new_time,
        },
    )

    return {
        "success": True,
        "rescheduled": {
            "id": appointment_id,
            "old_date": appt["start_time"][:10],
            "old_time": appt["start_time"][11:16],
            "new_date": new_date,
            "new_time": new_time,
            "type": appt["appointment_type"],
            "duration": duration,
        },
    }
