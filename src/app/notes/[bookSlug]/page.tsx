import Link from 'next/link';
import { notFound } from 'next/navigation';
import { getBookBySlug, getBooks, getNotesForChapter, notePath } from '@/lib/content';
import { pageMetadata } from '@/lib/metadata';

type Props = { params: { bookSlug: string } };

export function generateStaticParams() {
  return getBooks().map(book => ({ bookSlug: book.slug }));
}

export function generateMetadata({ params }: Props) {
  const book = getBookBySlug(params.bookSlug);
  if (!book) notFound();
  return pageMetadata(book.title, book.description, `/notes/${book.slug}`);
}

export default function BookIndex({ params }: Props) {
  const book = getBookBySlug(params.bookSlug);
  if (!book) notFound();
  return (
    <>
      <nav className="breadcrumbs" aria-label="Breadcrumb"><Link href="/notes">← All study notes</Link></nav>
      <header className="page-header">
        <p className="eyebrow">Textbook companion · {book.author}</p>
        <h1>{book.title}</h1>
        <p>{book.description}</p>
      </header>
      {book.chapters.map(chapter => {
        const notes = getNotesForChapter(book.slug, chapter.slug);
        return (
          <section key={chapter.id} className="chapter" aria-labelledby={chapter.slug}>
            <div className="chapter-heading">
              <span className="chapter-number">CH {String(chapter.id).padStart(2, '0')}</span>
              <h2 id={chapter.slug}>{chapter.title}</h2>
              {notes.length === 0 && <span className="chapter-status">Notes to come</span>}
            </div>
            {notes.length > 0 && <ul>{notes.map(note => (
              <li key={notePath(book.slug, note)}><Link href={notePath(book.slug, note)}>{note.title} <span aria-hidden="true">→</span></Link></li>
            ))}</ul>}
          </section>
        );
      })}
    </>
  );
}
