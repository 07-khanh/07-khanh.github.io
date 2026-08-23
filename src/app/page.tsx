import { getNoteBySlug, getAllNotes } from '@/lib/content';
import { notFound } from 'next/navigation';
import Link from 'next/link';

interface PageProps {
  params: {
    slug: string[];
  };
}

// Function to automatically extract headings from raw MDX text
function extractHeadings(content: string) {
  const headingLines = content.split('\n').filter((line) => line.match(/^#{1-3}\s+/));

  return headingLines.map((raw) => {
    const level = raw.match(/^#+/)?.[0].length || 1;
    const text = raw.replace(/^#+\s+/, '').trim();
    // Generate anchor slug matching standard Markdown rules
    const id = text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-');

    return { level, text, id };
  });
}

export default async function NotePage({ params }: PageProps) {
  const slugPath = params.slug ? params.slug.join('/') : '';
  const note = await getNoteBySlug(slugPath);

  if (!note) {
    notFound();
  }

  const headings = extractHeadings(note.content);

  return (
    <article className="prose prose-neutral max-w-none">
      <header className="mb-8 border-b border-neutral-200 pb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2">{note.meta.title}</h1>
        {note.meta.description && (
          <p className="text-neutral-600 text-lg">{note.meta.description}</p>
        )}
        <div className="flex gap-4 text-sm text-neutral-500 font-sans mt-4">
          <span>{note.meta.date}</span>
          {note.meta.chapter && <span>Chapter {note.meta.chapter}</span>}
        </div>
      </header>

      {/* AUTOMATIC TABLE OF CONTENTS */}
      {headings.length > 0 && (
        <nav className="my-8 p-6 bg-neutral-100/60 rounded-lg border border-neutral-200/80 font-sans">
          <h2 className="text-base font-semibold text-neutral-900 mb-3 mt-0">
            Table of Contents
          </h2>
          <ul className="space-y-1.5 text-sm list-none p-0 m-0">
            {headings.map((h, i) => (
              <li
                key={i}
                style={{ paddingLeft: `${(h.level - 1) * 1}rem` }}
                className="m-0"
              >
                <a
                  href={`#${h.id}`}
                  className="text-neutral-700 hover:text-blue-600 no-underline hover:underline"
                >
                  {h.text}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      )}

      {/* Render Note MDX Content */}
      <div className="mt-8">{note.component}</div>
    </article>
  );
}
