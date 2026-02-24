create extension if not exists pgcrypto;

create table if not exists public.reliability_leads (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  email text not null,
  team_size text,
  use_case text,
  source text not null default 'app.conch.so',
  created_at timestamptz not null default now()
);

create index if not exists reliability_leads_created_at_idx
  on public.reliability_leads (created_at desc);

-- Keep writes to server-side only (service role in Next API route).
revoke all on table public.reliability_leads from anon, authenticated;
