import Link from 'next/link';
import { getBookNotes, getBooks } from '@/lib/content';
import { pageMetadata } from '@/lib/metadata';

export const metadata = pageMetadata('Study notes', 'Personal derivations, intuition, and code experiments from studying machine learning.', '/notes');

export default function NotesIndex() {
  return (
    <>
      <header className="page-header">
        <p className="eyebrow">The learning notebook</p>
        <h1>Study notes</h1>
        <p>My derivations, intuition, and code experiments from reading. These are personal study notes, collected and refined as I learn.</p>
      </header>
      {getBooks().map(book => (
        <Link href={`/notes/${book.slug}`} key={book.slug} className="notebook">
          <span className="eyebrow">{getBookNotes(book.slug).length} notes · Textbook companion</span>
          <h2 className="mt-3">{book.title} <span aria-hidden="true">↗</span></h2>
          <p>By {book.author}</p>
          <p>{book.description}</p>
        </Link>
      ))}
    </>
  );
}
