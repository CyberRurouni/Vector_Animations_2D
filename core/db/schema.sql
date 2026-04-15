-- Enable pgvector extension
create extension if not exists vector;
 
-- 2. Create the Icons Table (The Polished Library)
create table assets (
    id  uuid primary key default gen_random_uuid(),
 
    -- What the scene planner asked for (used for semantic search)
    concept         text not null,
 
    -- snake_case label produced by the LLM prompt builder
    asset_name         text not null,
 
    -- Full image-generation prompt sent to Runware / Pollinations
    prompt          text not null,

    -- In which category it lies for example; sketch, icon
    asset_type text not null,
 
    -- Style tag produced by the LLM alongside the prompt
    -- e.g. "silhouette", "outline", "solid", "light_colored", "pencil-sketch"
    style_tag       text not null default 'silhouette',
 
    -- Embedding of: "Icon {name} is used for portraying concepts like: {concept}"
    embedding       vector(1536) not null,
 
    -- Where the processed PNG lives in Supabase Storage
    storage_url     text not null,
 
    -- Reuse tracking
    usage_count     integer not null default 0,
    last_used_at    timestamptz,
 
    -- Provider info, model used, dimensions, cost, etc.
    metadata        jsonb not null default '{}'::jsonb,
 
    created_at      timestamptz not null default now()
);
 

-- Create HNSW indexes for fast vector similarity search
create index on public.assets using hnsw (embedding vector_cosine_ops);

-- RPC: Increment_icon_usage
create or replace function increment_asset_usage(p_asset_id uuid)
returns void
language sql
as $$
    update public.assets
    set
        usage_count  = usage_count + 1,
        last_used_at = now()
    where id = p_asset_id;
$$;

-- RPC: Match Asset
create or replace function match_asset(
    query_embedding vector(1536),
    p_asset_type text,
    p_style_tag text default null,
    similarity_threshold float default 0.9
)
returns table (
    id uuid,
    storage_url text,
    similarity float
)
language sql
as $$
    select
        id,
        storage_url,
        1 - (embedding <=> query_embedding) as similarity
    from public.assets
    where
        asset_type = p_asset_type
        and (
            p_style_tag is null
            or style_tag = p_style_tag
        )
        and (embedding <=> query_embedding) <= (1 - similarity_threshold)
    order by embedding <=> query_embedding
    limit 1;
$$;

-- Enable Row Level Security (RLS) 
alter table public.assets enable row level security;

-- Create a policy that allows you (authenticated user) to do everything
create policy "Allow all access to authenticated users" 
on public.assets for all 
to authenticated 
using (true);