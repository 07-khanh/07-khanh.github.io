import Link from 'next/link';
import { renderMarkdown } from '@/lib/markdown';
import { formatDate } from '@/lib/site';

interface ArticleLink { href: string; title: string }
interface Props {
  title: string;
  description: string;
  date: string;
  content: string;
  breadcrumbs: ArticleLink[];
  previous?: ArticleLink;
  next?: ArticleLink;
}

export async function Article({ title, description, date, content, breadcrumbs, previous, next }: Props) {
  const rendered = await renderMarkdown(content);
  const headings = rendered.headings.filter(heading => heading.level <= 3);
  return (
    <article className="article">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link href="/">Home</Link>
        {breadcrumbs.map(crumb => <span key={crumb.href}><span aria-hidden="true"> / </span><Link href={crumb.href}>{crumb.title}</Link></span>)}
      </nav>
      <header className="page-header">
        <p className="eyebrow">{breadcrumbs[0]?.title}</p>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
        <div className="article-meta"><time dateTime={date}>{formatDate(date)}</time><span>About {rendered.readingMinutes} min read</span></div>
      </header>
      {headings.length >= 4 && (
        <details className="contents">
          <summary>On this page · {headings.length} sections</summary>
          <nav aria-label="Table of contents"><ol>{headings.map(heading => (
            <li key={heading.id} className={heading.level > 2 ? 'subheading' : undefined}>
              <a href={`#${heading.id}`}>{heading.text}</a>
            </li>
          ))}</ol></nav>
        </details>
      )}
      <div className="prose" dangerouslySetInnerHTML={{ __html: rendered.html }} />
      {(previous || next) && <nav className="article-navigation" aria-label="Continue reading">
        <div>{previous && <Link href={previous.href}><span>← Previous</span>{previous.title}</Link>}</div>
        <div>{next && <Link href={next.href}><span>Next →</span>{next.title}</Link>}</div>
      </nav>}
    </article>
  );
}
