import { getBookBySlug, getBooks, getNoteContent, getNotesForChapter } from '@/lib/content';
import { markdownToHtml } from '@/lib/markdown';
import Link from 'next/link';
import { notFound } from 'next/navigation';

export async function generateStaticParams() {
  const params: any[] = [];
  const books = getBooks();
  for (const book of books) {
    if (!book) continue;
    for (const chapter of book.chapters) {
      if (chapter.hasNotes) {
        const notes = getNotesForChapter(book.slug, chapter.slug);
        for (const note of notes) {
          params.push({
            bookSlug: book.slug,
            chapterSlug: chapter.slug,
            noteSlug: note.slug
          });
        }
      }
    }
  }
  return params;
}

export default async function NotePage({ params }: { params: { bookSlug: string, chapterSlug: string, noteSlug: string } }) {
  const book = getBookBySlug(params.bookSlug);
  const note = getNoteContent(params.bookSlug, params.chapterSlug, params.noteSlug);
  
  if (!book || !note) notFound();
  
  const chapter = book.chapters.find((c: any) => c.slug === params.chapterSlug);
  const html = await markdownToHtml(note.content);
  
  return (
    <article className="prose max-w-none font-serif text-[1.125rem]">
      <nav className="not-prose mb-8 text-sm font-sans text-gray-500 flex items-center gap-2">
        <Link href={`/notes/${book.slug}`} className="hover:text-gray-900 transition-colors">{book.title}</Link>
        <span>/</span>
        <span className="text-gray-900 font-medium">{chapter?.title}</span>
      </nav>
      
      <header className="mb-12 not-prose border-b border-gray-200 pb-8">
        <h1 className="text-4xl font-sans font-bold text-gray-900 leading-tight">{note.meta.title}</h1>
        {note.meta.description && (
          <p className="text-xl text-gray-500 mt-4 leading-relaxed font-serif">{note.meta.description}</p>
        )}
      </header>
      
      <div dangerouslySetInnerHTML={{ __html: html }} />
      
      <footer className="not-prose mt-20 pt-8 border-t border-gray-200">
        <Link href={`/notes/${book.slug}`} className="text-amber-600 font-sans font-medium hover:underline">
          &larr; Back to {book.title}
        </Link>
      </footer>
    </article>
  );
}
