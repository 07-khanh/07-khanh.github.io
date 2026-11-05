import { notFound } from 'next/navigation';
import { Article } from '@/components/Article';
import { getBookBySlug, getBookNotes, getBooks, getNoteContent, notePath } from '@/lib/content';
import { pageMetadata } from '@/lib/metadata';

type Props = { params: { bookSlug: string; chapterSlug: string; noteSlug: string } };

export function generateStaticParams() {
  return getBooks().flatMap(book => getBookNotes(book.slug).map(note => ({
    bookSlug: book.slug, chapterSlug: note.chapterSlug, noteSlug: note.slug,
  })));
}

export function generateMetadata({ params }: Props) {
  const note = getNoteContent(params.bookSlug, params.chapterSlug, params.noteSlug);
  if (!note) notFound();
  return pageMetadata(note.meta.title, note.meta.description,
    `/notes/${params.bookSlug}/${params.chapterSlug}/${params.noteSlug}`, note.meta.date);
}

export default async function NotePage({ params }: Props) {
  const book = getBookBySlug(params.bookSlug);
  const note = getNoteContent(params.bookSlug, params.chapterSlug, params.noteSlug);
  if (!book || !note) notFound();
  const chapter = book.chapters.find(item => item.id === note.meta.chapter || (!note.meta.chapter && item.slug === params.chapterSlug));
  const notes = getBookNotes(book.slug);
  const index = notes.findIndex(item => item.chapterSlug === params.chapterSlug && item.slug === params.noteSlug);
  const link = (item: typeof notes[number] | undefined) => item ? { href: notePath(book.slug, item), title: item.title } : undefined;
  return <Article {...note.meta} content={note.content}
    breadcrumbs={[
      { href: '/notes', title: 'Study notes' },
      { href: `/notes/${book.slug}`, title: book.title },
      ...(chapter ? [{ href: `/notes/${book.slug}#${chapter.slug}`, title: `Chapter ${chapter.id}` }] : []),
    ]}
    previous={link(notes[index - 1])} next={link(notes[index + 1])} />;
}
