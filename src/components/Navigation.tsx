'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { site } from '@/lib/site';

const links = [
  { href: '/', label: 'About' },
  { href: '/projects', label: 'Projects' },
  { href: '/blog', label: 'Writing' },
  { href: '/notes', label: 'Notes' },
];

export function Navigation() {
  const pathname = usePathname();
  return (
    <header className="site-header">
      <div className="shell header-inner">
        <Link href="/" className="wordmark" aria-label={`${site.name}, home`}>Khanh<span aria-hidden="true">.</span></Link>
        <nav aria-label="Main navigation" className="main-nav">
          {links.map(({ href, label }) => {
            const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
            return <Link key={href} href={href} aria-current={active ? 'page' : undefined}>{label}</Link>;
          })}
        </nav>
      </div>
    </header>
  );
}
