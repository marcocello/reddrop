from pathlib import Path


def test_threads_table_starts_with_id_title_social_and_job_columns() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    id_index = source.find("title='ID'")
    title_index = source.find("title='Title'")
    social_index = source.find("title='Social'")
    job_index = source.find("title='Job'")

    assert id_index != -1
    assert title_index != -1
    assert social_index != -1
    assert job_index != -1
    assert id_index < title_index < social_index < job_index

    assert 'shortId(row.original.conversation_id)' in source
    assert "title='Job'" in source
    assert "{row.original.job_name}" in source
    assert "className='leading-tight'" in source
    assert "truncate text-sm font-medium" in source
    assert "block font-mono text-xs text-muted-foreground" in source


def test_threads_table_does_not_include_score_column() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "title='Score'" not in source
    assert "accessorKey: 'score'" not in source


def test_threads_social_column_uses_badge_with_brand_icon() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'" in source
    assert "import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'" in source
    assert 'function socialIcon(platform: ThreadItem[\'social\']): IconDefinition {' in source
    assert "<Badge className='bg-muted text-muted-foreground'>" in source
    assert "<FontAwesomeIcon icon={socialIcon(row.original.social)} className='size-3.5' />" in source
    assert "{formatSocial(row.original.social)}" in source


def test_threads_page_heading_is_conversations() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert '>Conversations<' in source
    assert '>Threads<' not in source


def test_threads_search_placeholder_mentions_job_id() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert 'Search by thread, job id, subreddit or URL...' in source

def test_thread_sidebar_url_is_clamped_to_two_lines() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert 'WebkitLineClamp: 2' in source
    assert "parts.join('\\n')" not in source

def test_thread_cell_only_contains_two_line_title() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "{row.original.job_name} · r/{row.original.subreddit}" not in source
    assert "block max-w-[28rem] line-clamp-2 font-medium leading-snug" in source


def test_select_and_id_columns_are_pinned_first_in_column_order_logic() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "const SELECT_COLUMN_ID = 'select'" in source
    assert "const ID_COLUMN_ID = 'id'" in source
    assert "next = next.filter((id) => id !== ID_COLUMN_ID && id !== SELECT_COLUMN_ID)" in source
    assert "next.unshift(SELECT_COLUMN_ID)" in source
    assert "next.unshift(ID_COLUMN_ID)" in source


def test_threads_toolbar_filters_order_and_no_subreddit_column() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    social_filter_index = source.find("columnId: 'social', title: 'Social'")
    job_filter_index = source.find("columnId: 'job', title: 'Job'")
    status_filter_index = source.find("columnId: 'status'")

    assert social_filter_index != -1
    assert job_filter_index != -1
    assert status_filter_index != -1
    assert social_filter_index < job_filter_index < status_filter_index

    assert "columnId: 'subreddit', title: 'Subreddit'" not in source
    assert "title='Subreddit'" not in source


def test_threads_actions_include_delete_conversation_with_confirmation() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "const SELECT_COLUMN_ID = 'select'" in source
    assert "Select all visible conversations" in source
    assert "deleteLabel='Delete selected'" in source
    assert "requestDeleteSelectedThreads()" in source
    assert "const [selectedConversationIds, setSelectedConversationIds] = useState<Set<string>>(new Set())" in source
    assert "const [pendingDeleteThreads, setPendingDeleteThreads] = useState<ThreadItem[]>([])" in source
    assert 'reddropApi.deleteThread(' in source
    assert "title='Delete conversation'" in source
    assert "confirmText='Delete'" in source
    assert "className='gap-2 text-destructive'" in source
    assert 'requestDeleteThread(thread)' in source


def test_threads_date_format_supports_date_only_linkedin_values() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "if (!value) return 'ND'" in source
    assert "if (typeof value === 'string') {" in source
    assert "const trimmed = value.trim()" in source
    assert "if (/^\\d{4}-\\d{2}-\\d{2}$/.test(trimmed)) return trimmed" in source
    assert "toISOString().replace('T', ' ').slice(0, 16)" in source


def test_threads_compose_actions_use_generate_send_reply_with_loading_and_fresh_generation() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    assert "<DropdownMenuItem onSelect={() => void performReply(thread)}>Generate Reply</DropdownMenuItem>" in source
    assert 'Send Reply' in source
    assert 'Generate Reply' in source
    assert "import { ChevronDown, Loader2, MoreHorizontal, RefreshCw } from 'lucide-react'" in source
    assert "selectedReplyBusy ? <Loader2 className='size-4 animate-spin' />" in source
    assert "selectedSendBusy ? <Loader2 className='size-4 animate-spin' />" in source
    assert "replyDraft.length > 0 ? replyDraft : undefined" not in source
    assert "const replyDraft = replyInput.trim()" not in source


def test_threads_manual_reply_edits_are_debounced_and_autosaved() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    assert "const [replyAutosaveState, setReplyAutosaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')" in source
    assert "sheetTab !== 'compose'" in source
    assert "const draftReply = replyInput.trim()" in source
    assert "const persistedReply = (selectedThread.reply ?? '').trim()" in source
    assert "const autosaveTimer = window.setTimeout(() => {" in source
    assert "void persistReplyDraft(selectedThread, replyInput, { notifyOnError: false })" in source
    assert "window.clearTimeout(autosaveTimer)" in source
    assert "Autosaving..." in source
    assert "Saved automatically" in source
