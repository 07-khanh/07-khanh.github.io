import { getPostBySlug, getPosts } from '@/lib/content';
import { notFound } from 'next/navigation';

// Flexible regex to extract any h1-h3 heading from rendered HTML
function extractHeadingsFromHtml(html: string) {
  // Matches any <h1 ...>, <h2 ...>, or <h3 ...> tag with any attributes
  const regex = /<h([1-3])\b([^>]*)>([\s\S]*?)<\/h\1>/gi;
  const headings = [];
  let match;

  while ((match = regex.exec(html)) !== null) {
    const level = parseInt(match[1], 10);
    const attributes = match[2];
    const rawContent = match[3];

    // Extract existing id attribute if present (e.g. id="my-heading")
    const idMatch = attributes.match(/id=["']([^"']+)["']/i);

    // Strip nested tags (like <a>, <code>, <span>, math) to get clean text
    const text = rawContent.replace(/<[^>]+>/g, '').trim();

    // Fallback to slugifying text if no ID attribute exists on the header
    const id =
      idMatch?.[1] ||
      text
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-');

    // Skip page titles or empty headers if needed
    if (text) {
      headings.push({ level, text, id });
    }
  }

  return headings;
}

export default async function NotePage() {
  const posts = await getPosts();
  const defaultSlug = posts[0]?.slug || 'index';
  const note = await getPostBySlug(defaultSlug);

  if (!note) {
    notFound();
  }

  const headings = extractHeadingsFromHtml(note.content);

  return (
    <article className="max-w-none font-serif">
      <header className="mb-8 border-b border-neutral-200 pb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2 text-[#1a1a1a]">
          {note.meta.title}
        </h1>
        {note.meta.description && (
          <p className="text-neutral-600 text-lg">{note.meta.description}</p>
        )}
        <div className="flex gap-4 text-sm text-neutral-500 font-sans mt-4">
          {note.meta.date && <span>{note.meta.date}</span>}
          {note.meta.chapter && <span>Chapter {note.meta.chapter}</span>}
        </div>
      </header>

      {/* TABLE OF CONTENTS */}
      {headings.length > 0 && (
        <nav className="my-8 p-8 bg-[#f3f1ec] rounded-xl">
          <h2 className="text-lg font-bold text-[#1a1a1a] mb-4 mt-0 border-none p-0">
            Contents
          </h2>
          <ol className="space-y-3 list-decimal list-inside text-[16px] p-0 m-0 text-[#1a1a1a] font-bold">
            {headings.map((h, i) => (
              <li
                key={i}
                style={{ paddingLeft: `${(h.level - 1) * 0.75}rem` }}
                className="m-0"
              >
                <a
                  href={`#${h.id}`}
                  className="text-[#2563eb] hover:underline font-normal inline-ml-1"
                >
                  {h.text}
                </a>
              </li>
            ))}
          </ol>
        </nav>
      )}

      {/* POST CONTENT */}
      <div
        className="mt-8 prose prose-neutral max-w-none text-[17px] leading-relaxed"
        dangerouslySetInnerHTML={{ __html: note.content }}
      />
    </article>
  );
}
