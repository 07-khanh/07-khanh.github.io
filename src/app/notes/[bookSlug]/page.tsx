import { getBookBySlug, getBooks, getNotesForChapter } from '@/lib/content';
import Link from 'next/link';
import { notFound } from 'next/navigation';

export async function generateStaticParams() {
  return getBooks().map(b => ({ bookSlug: b?.slug }));
}

export default function BookIndex({ params }: { params: { bookSlug: string } }) {
  const book = getBookBySlug(params.bookSlug);
  if (!book) notFound();

  return (
    <div className="space-y-12">
      <header className="border-b border-gray-200 pb-8">
        <Link href="/notes" className="text-sm font-sans text-amber-600 hover:underline mb-6 inline-block">&larr; All Notebooks</Link>
        <h1 className="text-3xl font-sans font-bold text-gray-900 leading-tight">{book.title}</h1>
        <p className="text-gray-500 mt-3 font-sans font-medium uppercase tracking-wide">By {book.author}</p>
      </header>
      
      <div className="space-y-10">
        {book.chapters.map((chapter: any) => {
          const notes = chapter.hasNotes ? getNotesForChapter(book.slug, chapter.slug) : [];
          return (
            <section key={chapter.id} className="font-sans">
              <h2 className="text-lg font-bold text-gray-900 flex items-center gap-3">
                <span className="text-gray-400 font-mono text-sm w-12 shrink-0">CH {chapter.id.toString().padStart(2, '0')}</span>
                {chapter.title}
                {!chapter.hasNotes && <span className="text-[0.7rem] bg-gray-100 text-gray-500 px-2 py-0.5 rounded-md font-bold uppercase tracking-wider ml-2">Pending</span>}
              </h2>
              {chapter.hasNotes && notes.length > 0 && (
                <ul className="mt-4 ml-16 space-y-3">
                  {notes.map(note => (
                    <li key={note.slug}>
                      <Link href={`/notes/${book.slug}/${chapter.slug}/${note.slug}`} className="text-gray-600 font-serif text-lg hover:text-amber-600 transition-colors">
                        {note.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
